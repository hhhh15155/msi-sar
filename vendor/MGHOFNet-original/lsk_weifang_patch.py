import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from operator import truediv
import time, os, pandas as pd
from tqdm import tqdm
import model as model_pkg
import dataset

# ================= 配置区域 =================
PATCH_SIZES_TO_TEST = [5, 7, 9, 11, 13, 15]
EPOCHS = 200
BATCH_SIZE = 64
RANDOM_SEED = 42
PCA_COMPONENTS = 30
TRAIN_SAMPLES_PER_CLASS = 10
NUM_CLASSES = 8
SAVE_DIR_ROOT = 'cls_result/lsk_patch_study_weifang'


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def loadData():
    data_HSI = sio.loadmat('data/weifang/hsi.mat')['image_data']
    data_lidar = sio.loadmat('data/weifang/lidar.mat')['image_data']
    labels = sio.loadmat('data/weifang/label.mat')['image_data']
    return data_HSI, data_lidar, labels


def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX


def select_traintest(groundTruth, num_train_per_class):
    train, test = {}, {}
    m = int(max(groundTruth))
    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indices)
        nb_val = num_train_per_class
        if len(indices) < nb_val: nb_val = int(len(indices) * 0.5)
        train[i] = indices[:nb_val]
        test[i] = indices[nb_val:]
    train_indices, test_indices = [], []
    for i in range(m):
        train_indices += train[i];
        test_indices += test[i]
    np.random.shuffle(train_indices);
    np.random.shuffle(test_indices)
    return train_indices, test_indices


def run_experiment(patch_size, X1_pca, X2_raw, y_gt, train_idx, test_idx, device):
    print(f'\n>>> Patch Size: {patch_size}')
    # 关键：使用当前 Patch Size Padding，但基于固定的 PCA 特征和索引
    X1_p = dataset.pad_and_normalize(X1_pca, patch_size, mode='hsi')
    X2_p = dataset.pad_and_normalize(X2_raw, patch_size, mode='lidar')

    train_loader = dataset.get_dataloader(X1_p, X2_p, y_gt, train_idx, BATCH_SIZE, patch_size, shuffle=True,
                                          cache_data=True)
    test_loader = dataset.get_dataloader(X1_p, X2_p, y_gt, test_idx, BATCH_SIZE, patch_size)

    model = model_pkg.TTTFusionNet(PCA_COMPONENTS, 1, NUM_CLASSES, 128, 2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()  # 无权重对齐

    best_oa = 0
    for epoch in range(EPOCHS):
        model.train()
        for hsi, lidar, labels in train_loader:
            hsi, lidar, labels = hsi.to(device), lidar.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(hsi, lidar)
            criterion(logits, labels).backward()
            optimizer.step()
        scheduler.step()

        if (epoch + 1) % 10 == 0 or (epoch + 1) == EPOCHS:
            model.eval()
            y_p, y_t = [], []
            with torch.no_grad():
                for h, l, lab in test_loader:
                    y_p.extend(model(h.to(device), l.to(device)).max(1)[1].cpu().numpy())
                    y_t.extend(lab.numpy())
            oa = accuracy_score(y_t, y_p) * 100
            if oa > best_oa: best_oa = oa

    return {'Patch Size': patch_size, 'OA': best_oa}


def main():
    os.makedirs(SAVE_DIR_ROOT, exist_ok=True)
    set_seed(RANDOM_SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. 循环外固定 PCA 和样本索引以对齐原始实验
    X1_raw, X2_raw, y_raw = loadData()
    if X2_raw.ndim == 2: X2_raw = X2_raw[:, :, np.newaxis]
    X1_pca = applyPCA(X1_raw, PCA_COMPONENTS)
    gt = y_raw.flatten().astype(int)
    train_idx, test_idx = select_traintest(gt, TRAIN_SAMPLES_PER_CLASS)

    results = []
    for p in PATCH_SIZES_TO_TEST:
        results.append(run_experiment(p, X1_pca, X2_raw, y_raw, train_idx, test_idx, device))

    df = pd.DataFrame(results)
    print("\nSUMMARY:\n", df)
    df.to_csv(f'{SAVE_DIR_ROOT}/lsk_patch_summary.csv', index=False)


if __name__ == '__main__':
    main()