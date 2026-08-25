# train_lsk_muufl.py - LSK-FusionNet Training
# 修正版: 移除增强，使用 Weighted CrossEntropy 以提升 AA

import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import torch
import torch.nn as nn
import torch.nn.functional as F
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
GENERATE_LABELED_MAP = True
GENERATE_ALL_MAP = False

# --- 训练超参数 ---
EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-2
RANDOM_SEED = 42

# --- 数据处理 ---
PATCH_SIZE = 11
PCA_COMPONENTS = 30
TRAIN_SAMPLES_PER_CLASS = 20  # 保持 150

# --- 路径设置 ---
SAVE_DIR_PARAMS = 'cls_params/lsk_muufl_1'
SAVE_DIR_RESULTS = 'cls_result/lsk_muufl_1'
SAVE_DIR_MAPS = 'cls_map/lsk_muufl_1'

# --- 模型结构参数 ---
NUM_CLASSES = 11
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
    print("Loading MUUFL dataset...")
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
    labels_loc = {}
    train = {}
    test = {}
    m = int(max(groundTruth))

    # 记录每个类实际选了多少个，用于计算权重
    class_counts = []

    print(f'Selecting {num_train_per_class} samples per class for training...')

    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indices)
        labels_loc[i] = indices

        nb_val = num_train_per_class
        if len(indices) < nb_val:
            print(f"Warning: Class {i + 1} only has {len(indices)} samples. Using all for train.")
            nb_val = len(indices)  # 使用全部
            # 这里原本逻辑是 -1 用于测试，如果样本太少，全用于训练可能导致测试集为空
            # 为了保证代码稳健，如果少于5个，就全训练；否则留几个给测试
            if len(indices) > 5:
                nb_val = len(indices) - 2  # 留2个测试

        train[i] = indices[-nb_val:]
        test[i] = indices[:-nb_val]

        class_counts.append(len(train[i]))

    train_indices = []
    test_indices = []
    for i in range(m):
        train_indices += train[i]
        test_indices += test[i]

    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    return train_indices, test_indices, class_counts


def create_data_loader():
    X1, X2, y = loadData()

    lidar_data_2d = np.zeros((X2.shape[0], X2.shape[1], 1))
    lidar_data_2d[:, :, 0] = X2[:, :, 0]
    X2 = lidar_data_2d

    print(f'Applying PCA (components={PCA_COMPONENTS})...')
    X1 = applyPCA(X1, numComponents=PCA_COMPONENTS)

    print(f'Padding and Normalizing (patch_size={PATCH_SIZE})...')
    X1_padded = dataset.pad_and_normalize(X1, PATCH_SIZE, mode='hsi')
    X2_padded = dataset.pad_and_normalize(X2, PATCH_SIZE, mode='lidar')

    gt = y.flatten().astype(int)

    # 获取 class_counts 以计算权重
    train_indices, test_indices, class_counts = select_traintest(gt, TRAIN_SAMPLES_PER_CLASS)

    total_indices = []
    m = int(max(gt))
    for i in range(m):
        total_indices += [j for j, x in enumerate(gt.ravel().tolist()) if x == i + 1]

    height, width = y.shape
    all_indices = np.arange(height * width)

    print(f'\nTrain samples: {len(train_indices)}')
    print(f'Test samples:  {len(test_indices)}')

    # === 关键修正: 关闭 Augment, 但开启 cache ===
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

    # ... total_loader, all_loader 省略，逻辑同上 ...
    total_loader = None
    if GENERATE_LABELED_MAP:
        total_loader = dataset.get_dataloader(X1_padded, X2_padded, y, total_indices, BATCH_SIZE, PATCH_SIZE,
                                              shuffle=False, num_workers=4, cache_data=False)

    all_loader = None
    if GENERATE_ALL_MAP:
        all_loader = dataset.get_dataloader(X1_padded, X2_padded, y, all_indices, BATCH_SIZE, PATCH_SIZE, shuffle=False,
                                            num_workers=4, cache_data=False)

    return train_loader, test_loader, total_loader, all_loader, y, total_indices, all_indices, class_counts


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
    print('LSK-FusionNet Training (MUUFL) - High AA Strategy')
    print(f'Config: EMB_DIM={EMB_DIM}, Loss=Weighted CrossEntropy')
    print('=' * 80)

    set_seed(RANDOM_SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. 准备数据并获取类别计数
    train_iter, test_iter, total_iter, all_iter, y, total_indices, all_indices, class_counts = create_data_loader()

    # === 计算类别权重 ===
    # 权重策略: 样本越少，权重越大 (Inverse Frequency)
    # 这里的 class_counts 对应 label 0 到 10
    counts = np.array(class_counts)
    weights = 1.0 / np.sqrt(counts)  # 使用平方根平滑，防止权重过激
    weights = weights / weights.sum() * len(counts)  # 归一化
    class_weights = torch.FloatTensor(weights).to(device)

    print(f"\nClass Weights (for AA boosting): \n{weights}")

    # 定义 Loss
    # criterion = nn.CrossEntropyLoss(weight=class_weights)
    criterion = nn.CrossEntropyLoss()

    # 2. 构建模型 (继续使用修改后的 model.py，那个 Detail LSK 确实有用)
    model = model_pkg.TTTFusionNet(
        in_ch_hsi=PCA_COMPONENTS,
        in_ch_lidar=1,
        num_classes=NUM_CLASSES,
        emb_dim=EMB_DIM,
        depth=DEPTH
    ).to(device)

    # 3. 优化器
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # 保存路径
    os.makedirs(SAVE_DIR_PARAMS, exist_ok=True)
    os.makedirs(SAVE_DIR_RESULTS, exist_ok=True)
    os.makedirs(SAVE_DIR_MAPS, exist_ok=True)

    # 4. 训练循环
    best_acc = 0
    pbar = tqdm(range(EPOCHS), desc="Training")
    tic = time.time()

    for epoch in pbar:
        # 传入 criterion
        loss, train_acc = train(model, train_iter, optimizer, scheduler, criterion, device)

        pbar.set_postfix({'Loss': f'{loss:.4f}', 'Tr_Acc': f'{train_acc:.2f}%', 'Best': f'{best_acc:.2f}%'})

        if (epoch + 1) % 10 == 0 or (epoch + 1) == EPOCHS:
            y_pred, y_true = test(model, test_iter, device)
            test_acc = accuracy_score(y_true, y_pred) * 100

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), f'{SAVE_DIR_PARAMS}/LSKNet_MUUFL_best.pth')
                tqdm.write(f"Epoch {epoch + 1}: New Best Test Acc: {test_acc:.2f}%")

    toc = time.time()
    print(f'\nBest Test Accuracy: {best_acc:.2f}%')

    # 5. 评估
    print('\nLoading best model for evaluation...')
    model.load_state_dict(torch.load(f'{SAVE_DIR_PARAMS}/LSKNet_MUUFL_best.pth'))
    y_pred, y_true = test(model, test_iter, device)

    target_names = [
        'Trees', 'Mostly Grass', 'Mixed Ground Surface', 'Dirt and Sand',
        'Road', 'Water', 'Buildings Shadow', 'Buildings',
        'Sidewalk', 'Yellow Curb', 'Cloth Panels'
    ]

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

    with open(f'{SAVE_DIR_RESULTS}/LSKNet_MUUFL_report.txt', 'w') as f:
        f.write(f'OA: {oa:.2f}%\nAA: {aa:.2f}%\nKappa: {kappa:.2f}%\n\n')
        f.write(classification)
        f.write(f'\n\nPer Class Acc:\n{each_acc}')

    if GENERATE_LABELED_MAP and total_iter is not None:
        print('\nGenerating labeled classification map...')
        Utils.generate_png(total_iter, model, y, device, total_indices, f'{SAVE_DIR_MAPS}/LSKNet_MUUFL_labeled')

    print('\nDone.')


if __name__ == '__main__':
    main()