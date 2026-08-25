# lsk_muufl_patch.py - MUUFL Patch Size Study for LSK-FusionNet
# 修正版: 无权重 Loss, 循环外固定样本索引与 PCA

import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from operator import truediv
import time
from tqdm import tqdm

# 导入核心模块
import model as model_pkg
import dataset
import Utils

# ================= 配置区域 =================
PATCH_SIZES_TO_TEST = [5, 7, 9, 11, 13, 15]

EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-2
RANDOM_SEED = 42

# 原始实验参数
PCA_COMPONENTS = 30
TRAIN_SAMPLES_PER_CLASS = 20
NUM_CLASSES = 11

SAVE_DIR_ROOT = 'cls_result/lsk_patch_study_muufl'


# ===========================================

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def loadData():
    """加载 MUUFL 数据集 """
    data_HSI = sio.loadmat('data/MUUFL/muufl.mat')['muufl']
    data_lidar = sio.loadmat('data/MUUFL/lidar_1.mat')['lidar']
    labels = sio.loadmat('data/MUUFL/muufl_label.mat')['muufl_gt']
    return data_HSI, data_lidar, labels


def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX


def select_traintest(groundTruth, num_train_per_class):
    """
    固定索引选择逻辑，对齐 lsk_muufl.py
    """
    train = {}
    test = {}
    m = int(max(groundTruth))
    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indices)
        nb_val = num_train_per_class
        if len(indices) < nb_val:
            nb_val = len(indices)
            if len(indices) > 5:
                nb_val = len(indices) - 2
        train[i] = indices[-nb_val:]
        test[i] = indices[:-nb_val]

    train_indices = []
    test_indices = []
    for i in range(m):
        train_indices += train[i]
        test_indices += test[i]
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    return train_indices, test_indices


def train_epoch(model, train_loader, optimizer, scheduler, criterion, device):
    model.train()
    for hsi, lidar, labels in train_loader:
        hsi, lidar, labels = hsi.to(device), lidar.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(hsi, lidar)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
    scheduler.step()


def test_epoch(model, test_loader, device):
    model.eval()
    y_pred, y_test = [], []
    with torch.no_grad():
        for hsi, lidar, labels in test_loader:
            hsi, lidar = hsi.to(device), lidar.to(device)
            outputs = model(hsi, lidar)
            _, predicted = outputs.max(1)
            y_pred.extend(predicted.cpu().numpy())
            y_test.extend(labels.numpy())
    return np.array(y_pred), np.array(y_test)


def main():
    os.makedirs(SAVE_DIR_ROOT, exist_ok=True)
    set_seed(RANDOM_SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. 关键：循环外固定数据加载、PCA 和样本索引选择，确保对齐原始实验
    X1_raw, X2_raw, y_raw = loadData()
    # LiDAR 维度调整 (H, W) -> (H, W, 1)
    X2_2d = X2_raw[:, :, 0:1] if X2_raw.ndim > 2 else X2_raw[:, :, np.newaxis]

    print(f'Applying global PCA (components={PCA_COMPONENTS})...')
    X1_pca = applyPCA(X1_raw, PCA_COMPONENTS)

    gt = y_raw.flatten().astype(int)
    print(f'Selecting fixed {TRAIN_SAMPLES_PER_CLASS} samples/class...')
    train_idx, test_idx = select_traintest(gt, TRAIN_SAMPLES_PER_CLASS)

    results = []

    # 2. 遍历不同的 Patch Size
    for p in PATCH_SIZES_TO_TEST:
        print(f'\n{"=" * 50}\nTesting MUUFL Patch Size: {p}\n{"=" * 50}')

        # 使用当前 Patch 大小进行 Padding，但基础特征和索引是固定的
        X1_padded = dataset.pad_and_normalize(X1_pca, p, mode='hsi')
        X2_padded = dataset.pad_and_normalize(X2_2d, p, mode='lidar')

        train_loader = dataset.get_dataloader(
            X1_padded, X2_padded, y_raw, train_idx,
            batch_size=BATCH_SIZE, patch_size=p, shuffle=True, cache_data=True
        )
        test_loader = dataset.get_dataloader(
            X1_padded, X2_padded, y_raw, test_idx,
            batch_size=BATCH_SIZE, patch_size=p
        )

        # 构建模型 (使用 DEPTH=2, EMB_DIM=128 对齐主实验)
        model = model_pkg.TTTFusionNet(
            in_ch_hsi=PCA_COMPONENTS, in_ch_lidar=1, num_classes=NUM_CLASSES,
            emb_dim=128, depth=2
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        # 修正：无权重 CrossEntropy
        criterion = nn.CrossEntropyLoss()

        best_oa = 0
        pbar = tqdm(range(EPOCHS), desc=f"P={p}", leave=False)
        for epoch in pbar:
            train_epoch(model, train_loader, optimizer, scheduler, criterion, device)

            if (epoch + 1) % 10 == 0 or (epoch + 1) == EPOCHS:
                y_p, y_t = test_epoch(model, test_loader, device)
                oa = accuracy_score(y_t, y_p) * 100
                if oa > best_oa:
                    best_oa = oa
                pbar.set_postfix({'Best': f'{best_oa:.2f}%'})

        print(f'Finished Patch {p}: Best OA = {best_oa:.2f}%')

        # 记录汇总结果
        results.append({'Patch Size': p, 'OA': best_oa})
        # 实时保存，防止中断
        pd.DataFrame(results).to_csv(f'{SAVE_DIR_ROOT}/lsk_muufl_patch_summary.csv', index=False)

    print(f"\nAll experiments done! Results saved to {SAVE_DIR_ROOT}/lsk_muufl_patch_summary.csv")


if __name__ == '__main__':
    main()