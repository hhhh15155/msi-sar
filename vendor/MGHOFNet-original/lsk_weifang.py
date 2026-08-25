# train_lsk_weifang.py - LSK-FusionNet Training for Weifang Dataset
# 基于 lsk_muufl.py 修改，适配 Weifang 数据集 (8类, 均衡采样)

import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import torch
import torch.nn as nn
import torch.optim as optim
from operator import truediv
import time
from tqdm import tqdm
import os

# 导入核心模块
import model as model_pkg
import dataset
import Utils

# ==========================================================
#                      Configuration
# ==========================================================

# --- 功能开关 ---
GENERATE_LABELED_MAP = True  # 生成带标签的预测图
GENERATE_ALL_MAP = False  # 生成全图预测

# --- 训练超参数 ---
EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
RANDOM_SEED = 42

# --- 数据处理 ---
PATCH_SIZE = 11
PCA_COMPONENTS = 30
TRAIN_SAMPLES_PER_CLASS = 10  # Weifang 标准实验通常每类取 50 个

# --- 路径设置 ---
# 确保 data/weifang/ 下有 hsi.mat, lidar.mat, label.mat
SAVE_DIR_PARAMS = 'cls_params/lsk_weifang'
SAVE_DIR_RESULTS = 'cls_result/lsk_weifang'
SAVE_DIR_MAPS = 'cls_map/lsk_weifang'

# --- 模型结构参数 (Weifang 8 Classes) ---
NUM_CLASSES = 8
EMB_DIM = 128
DEPTH = 2


# ==========================================================
# 1. Data Utils
# ==========================================================

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def loadData():
    """加载 Weifang 数据集，参考 sagdf_weifang.py"""
    print("Loading Weifang dataset...")
    # 路径与 key 根据 sagdf_weifang.py 保持一致
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
    """
    Weifang 均衡采样策略：每类随机选取固定数量
    """
    labels_loc = {}
    train = {}
    test = {}
    m = int(max(groundTruth))  # 应该为 8

    print(f'Selecting {num_train_per_class} samples per class for training...')

    for i in range(m):
        # 查找类别 i+1 的索引
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indices)
        labels_loc[i] = indices

        nb_val = num_train_per_class
        # 如果样本不足，取 50% 作为训练
        if len(indices) < nb_val:
            print(f"Warning: Class {i + 1} only has {len(indices)} samples. Using 50% for train.")
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


def create_data_loader():
    X1, X2, y = loadData()

    # 处理 Weifang LiDAR 维度: (H, W) -> (H, W, 1)
    if X2.ndim == 2:
        X2 = X2[:, :, np.newaxis]

    print(f'HSI shape: {X1.shape}, LiDAR shape: {X2.shape}, Label shape: {y.shape}')

    print(f'Applying PCA (components={PCA_COMPONENTS})...')
    X1 = applyPCA(X1, numComponents=PCA_COMPONENTS)

    print(f'Padding and Normalizing (patch_size={PATCH_SIZE})...')
    X1_padded = dataset.pad_and_normalize(X1, PATCH_SIZE, mode='hsi')
    X2_padded = dataset.pad_and_normalize(X2, PATCH_SIZE, mode='lidar')

    gt = y.flatten().astype(int)

    # 划分训练/测试集
    train_indices, test_indices = select_traintest(gt, TRAIN_SAMPLES_PER_CLASS)

    # 获取用于绘图的索引
    total_indices = []
    m = int(max(gt))
    for i in range(m):
        total_indices += [j for j, x in enumerate(gt.ravel().tolist()) if x == i + 1]

    height, width = y.shape
    all_indices = np.arange(height * width)

    print(f'\nTrain samples: {len(train_indices)}')
    print(f'Test samples:  {len(test_indices)}')

    # 构建 DataLoader
    # LSK 原始设置: augment=False, cache_data=True
    train_loader = dataset.get_dataloader(
        X1_padded, X2_padded, y, train_indices,
        batch_size=BATCH_SIZE, patch_size=PATCH_SIZE, shuffle=True,
        num_workers=0, cache_data=True, augment=False
    )

    test_loader = dataset.get_dataloader(
        X1_padded, X2_padded, y, test_indices,
        batch_size=BATCH_SIZE, patch_size=PATCH_SIZE, shuffle=False,
        num_workers=4, cache_data=False, augment=False
    )

    total_loader = None
    if GENERATE_LABELED_MAP:
        total_loader = dataset.get_dataloader(X1_padded, X2_padded, y, total_indices,
                                              BATCH_SIZE, PATCH_SIZE, shuffle=False,
                                              num_workers=4, cache_data=False)

    all_loader = None
    if GENERATE_ALL_MAP:
        all_loader = dataset.get_dataloader(X1_padded, X2_padded, y, all_indices,
                                            BATCH_SIZE, PATCH_SIZE, shuffle=False,
                                            num_workers=4, cache_data=False)

    return train_loader, test_loader, total_loader, all_loader, y, total_indices, all_indices


def train(model, train_loader, optimizer, scheduler, criterion, device):
    """训练一个 Epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for hsi, lidar, labels in train_loader:
        hsi, lidar, labels = hsi.to(device), lidar.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(hsi, lidar)

        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    scheduler.step()
    return total_loss / len(train_loader), 100. * correct / total


def test(model, test_loader, device):
    model.eval()
    y_pred_test = []
    y_test = []

    with torch.no_grad():
        for hsi, lidar, labels in test_loader:
            hsi, lidar = hsi.to(device), lidar.to(device)
            outputs = model(hsi, lidar)
            _, predicted = outputs.max(1)
            y_pred_test.extend(predicted.cpu().numpy())
            y_test.extend(labels.numpy())

    return np.array(y_pred_test), np.array(y_test)


def main():
    print('=' * 80)
    print('LSK-FusionNet Training (Weifang)')
    print(f'Config: EMB_DIM={EMB_DIM}, Samples/Class={TRAIN_SAMPLES_PER_CLASS}')
    print('=' * 80)

    set_seed(RANDOM_SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. 准备数据
    train_iter, test_iter, total_iter, all_iter, y, total_indices, all_indices = create_data_loader()

    # 2. 定义 Loss
    # Weifang 数据集采样是均衡的(每类50个)，因此不需要加权 CrossEntropy
    criterion = nn.CrossEntropyLoss()

    # 3. 构建模型
    # model.py 中的 TTTFusionNet 支持动态 num_classes
    model = model_pkg.TTTFusionNet(
        in_ch_hsi=PCA_COMPONENTS,
        in_ch_lidar=1,
        num_classes=NUM_CLASSES,  # 8
        emb_dim=EMB_DIM,
        depth=DEPTH
    ).to(device)

    # 4. 优化器
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # 保存路径
    os.makedirs(SAVE_DIR_PARAMS, exist_ok=True)
    os.makedirs(SAVE_DIR_RESULTS, exist_ok=True)
    os.makedirs(SAVE_DIR_MAPS, exist_ok=True)

    # 5. 训练循环
    best_acc = 0
    pbar = tqdm(range(EPOCHS), desc="Training")

    for epoch in pbar:
        loss, train_acc = train(model, train_iter, optimizer, scheduler, criterion, device)

        pbar.set_postfix({'Loss': f'{loss:.4f}', 'Tr_Acc': f'{train_acc:.2f}%', 'Best': f'{best_acc:.2f}%'})

        # 每 10 个 epoch 或最后一个 epoch 测试一次
        if (epoch + 1) % 10 == 0 or (epoch + 1) == EPOCHS:
            y_pred, y_true = test(model, test_iter, device)
            test_acc = accuracy_score(y_true, y_pred) * 100

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), f'{SAVE_DIR_PARAMS}/LSKNet_Weifang_best.pth')
                tqdm.write(f"Epoch {epoch + 1}: New Best Test Acc: {test_acc:.2f}%")

    print(f'\nBest Test Accuracy: {best_acc:.2f}%')

    # 6. 评估
    print('\nLoading best model for evaluation...')
    model.load_state_dict(torch.load(f'{SAVE_DIR_PARAMS}/LSKNet_Weifang_best.pth'))
    y_pred, y_true = test(model, test_iter, device)

    # Weifang 8 类 Generic Names
    target_names = [f'Class {i + 1}' for i in range(NUM_CLASSES)]

    classification = classification_report(y_true, y_pred, digits=4, target_names=target_names)
    oa = accuracy_score(y_true, y_pred) * 100
    confusion = confusion_matrix(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred) * 100

    list_diag = np.diag(confusion)
    list_raw_sum = np.sum(confusion, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum)) * 100
    aa = np.mean(each_acc)

    print('\nClassification Report:')
    print(classification)
    print(f'OA: {oa:.2f}% | AA: {aa:.2f}% | Kappa: {kappa:.2f}%')

    # 写入结果文件
    with open(f'{SAVE_DIR_RESULTS}/LSKNet_Weifang_report.txt', 'w') as f:
        f.write(f'Weifang Dataset (8 Classes)\n')
        f.write(f'Train Samples/Class: {TRAIN_SAMPLES_PER_CLASS}\n\n')
        f.write(f'OA: {oa:.2f}%\nAA: {aa:.2f}%\nKappa: {kappa:.2f}%\n\n')
        f.write(classification)
        f.write(f'\n\nPer Class Acc:\n{each_acc}')
        f.write(f'\n\nConfusion Matrix:\n{confusion}')

    # 7. 绘图
    if GENERATE_LABELED_MAP and total_iter is not None:
        print('\nGenerating labeled classification map...')
        Utils.generate_png(total_iter, model, y, device, total_indices, f'{SAVE_DIR_MAPS}/LSKNet_Weifang_labeled')

    if GENERATE_ALL_MAP and all_iter is not None:
        print('\nGenerating full classification map...')
        Utils.generate_all_png(all_iter, model, y, device, all_indices, f'{SAVE_DIR_MAPS}/LSKNet_Weifang_all')

    print('\nDone.')


if __name__ == '__main__':
    main()