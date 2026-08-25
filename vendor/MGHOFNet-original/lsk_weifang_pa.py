import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import itertools
import time

# 导入核心模块
import model as model_pkg
import dataset

# ================= 配置区域 =================
SEARCH_EPOCHS = 200
BATCH_SIZE = 64
RANDOM_SEED = 42

# 搜索网格
SEARCH_PARAMS = {
    'depth': [1, 2, 3],
    'emb_dim': [64, 128, 256]
}
RESULTS_FILE = 'cls_result/lsk_weifang_param_search.csv'

# 数据固定参数
PATCH_SIZE = 11
PCA_COMPONENTS = 30
TRAIN_SAMPLES_PER_CLASS = 10
NUM_CLASSES = 8

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
    train = {}
    test = {}
    m = int(max(groundTruth))
    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indices)
        nb_val = num_train_per_class
        if len(indices) < nb_val:
            nb_val = int(len(indices) * 0.5)
        train[i] = indices[:nb_val]
        test[i] = indices[nb_val:]
    train_indices = []
    test_indices = []
    for i in range(m):
        train_indices += train[i]
        test_indices += test[i]
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    return train_indices, test_indices

def train(model, train_loader, optimizer, scheduler, criterion, device):
    model.train()
    for hsi, lidar, labels in train_loader:
        hsi, lidar, labels = hsi.to(device), lidar.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(hsi, lidar)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
    scheduler.step()

def test(model, test_loader, device):
    model.eval()
    y_pred, y_test = [], []
    with torch.no_grad():
        for hsi, lidar, labels in test_loader:
            hsi, lidar = hsi.to(device), lidar.to(device)
            outputs = model(hsi, lidar)
            y_pred.extend(outputs.max(1)[1].cpu().numpy())
            y_test.extend(labels.numpy())
    return np.array(y_pred), np.array(y_test)

def run_experiment():
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(RANDOM_SEED)

    # 1. 准备数据 (搜索过程只加载一次)
    X1, X2, y = loadData()
    if X2.ndim == 2: X2 = X2[:, :, np.newaxis]
    X1 = applyPCA(X1, numComponents=PCA_COMPONENTS)
    X1_padded = dataset.pad_and_normalize(X1, PATCH_SIZE, mode='hsi')
    X2_padded = dataset.pad_and_normalize(X2, PATCH_SIZE, mode='lidar')
    gt = y.flatten().astype(int)
    train_indices, test_indices = select_traintest(gt, TRAIN_SAMPLES_PER_CLASS)

    train_loader = dataset.get_dataloader(X1_padded, X2_padded, y, train_indices,
                                          batch_size=BATCH_SIZE, patch_size=PATCH_SIZE,
                                          shuffle=True, cache_data=True)
    test_loader = dataset.get_dataloader(X1_padded, X2_padded, y, test_indices,
                                         batch_size=BATCH_SIZE, patch_size=PATCH_SIZE)

    # 2. 准备搜索组合
    combinations = [dict(zip(SEARCH_PARAMS.keys(), v)) for v in itertools.product(*SEARCH_PARAMS.values())]
    results = []

    for idx, config in enumerate(combinations):
        d, e = config['depth'], config['emb_dim']
        print(f"\n[{idx + 1}/{len(combinations)}] Config: Depth={d}, EmbDim={e}")

        model = model_pkg.TTTFusionNet(PCA_COMPONENTS, 1, NUM_CLASSES, e, d).to(device)
        criterion = nn.CrossEntropyLoss() # 无权重对齐
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SEARCH_EPOCHS)

        best_oa = 0.0
        start_time = time.time()
        for epoch in range(SEARCH_EPOCHS):
            train(model, train_loader, optimizer, scheduler, criterion, device)
            if (epoch + 1) % 10 == 0 or (epoch + 1) == SEARCH_EPOCHS:
                y_p, y_t = test(model, test_loader, device)
                oa = accuracy_score(y_t, y_p) * 100
                if oa > best_oa: best_oa = oa

        duration = time.time() - start_time
        print(f"  -> Best OA: {best_oa:.2f}%")
        results.append({'depth': d, 'emb_dim': e, 'OA': best_oa, 'Time': duration})
        pd.DataFrame(results).to_csv(RESULTS_FILE, index=False)

if __name__ == '__main__':
    run_experiment()