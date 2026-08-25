# train_lsk_muufl_aux_focal.py - LSK-FusionNet Training (with aux_loss + Focal Loss)

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
import model2 as model_pkg
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
TRAIN_SAMPLES_PER_CLASS = 20

# --- 路径设置 ---
SAVE_DIR_PARAMS = 'cls_params/lsk_muufl_2'
SAVE_DIR_RESULTS = 'cls_result/lsk_muufl_2'
SAVE_DIR_MAPS = 'cls_map/lsk_muufl_2'

# --- 模型结构参数 ---
NUM_CLASSES = 11
EMB_DIM = 128
DEPTH = 2

# --- Focal Loss 超参数 ---
FOCAL_GAMMA = 2.0
USE_FOCAL_ALPHA = False   # True: 使用 class_weights 作为 alpha; False: 不用 alpha（推荐先 False）


# ==========================================================
# 0. Focal Loss
# ==========================================================
class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss.

    logits: [B, C]
    targets: [B] (0..C-1)

    gamma: focusing parameter
    alpha: None or Tensor[C] (per-class weight)
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean', eps=1e-8):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = alpha  # Tensor[C] or None
        self.reduction = reduction
        self.eps = float(eps)

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)               # [B,C]
        probs = log_probs.exp().clamp_min(self.eps)            # [B,C]

        targets = targets.long()
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # [B]
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)          # [B]

        loss = (1.0 - pt).pow(self.gamma) * (-log_pt)          # [B]

        if self.alpha is not None:
            at = self.alpha.gather(0, targets)                 # [B]
            loss = at * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


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

    class_counts = []
    print(f'Selecting {num_train_per_class} samples per class for training...')

    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indices)
        labels_loc[i] = indices

        nb_val = num_train_per_class
        if len(indices) < nb_val:
            print(f"Warning: Class {i + 1} only has {len(indices)} samples. Using all for train.")
            nb_val = len(indices)
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
    train_indices, test_indices, class_counts = select_traintest(gt, TRAIN_SAMPLES_PER_CLASS)

    total_indices = []
    m = int(max(gt))
    for i in range(m):
        total_indices += [j for j, x in enumerate(gt.ravel().tolist()) if x == i + 1]

    height, width = y.shape
    all_indices = np.arange(height * width)

    print(f'\nTrain samples: {len(train_indices)}')
    print(f'Test samples:  {len(test_indices)}')

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
        total_loader = dataset.get_dataloader(
            X1_padded, X2_padded, y, total_indices,
            BATCH_SIZE, PATCH_SIZE, shuffle=False,
            num_workers=4, cache_data=False
        )

    all_loader = None
    if GENERATE_ALL_MAP:
        all_loader = dataset.get_dataloader(
            X1_padded, X2_padded, y, all_indices,
            BATCH_SIZE, PATCH_SIZE, shuffle=False,
            num_workers=4, cache_data=False
        )

    return train_loader, test_loader, total_loader, all_loader, y, total_indices, all_indices, class_counts


# ==========================================================
# 2. Training / Testing
# ==========================================================

def _collect_aux_loss(model, ref_tensor: torch.Tensor):
    aux = ref_tensor.new_tensor(0.0)
    for m in model.modules():
        if hasattr(m, "aux_loss"):
            v = m.aux_loss
            if not torch.is_tensor(v):
                v = ref_tensor.new_tensor(float(v))
            else:
                v = v.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
            aux = aux + v
    return aux


def train(model, train_loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0.0
    total_cls = 0.0
    total_aux = 0.0
    correct = 0
    total = 0

    for hsi, lidar, labels in train_loader:
        hsi, lidar, labels = hsi.to(device), lidar.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(hsi, lidar)

        cls_loss = criterion(logits, labels)
        aux_loss = _collect_aux_loss(model, cls_loss)
        loss = cls_loss + aux_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_cls += cls_loss.item()
        total_aux += aux_loss.item()

        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    avg_cls = total_cls / len(train_loader)
    avg_aux = total_aux / len(train_loader)
    acc = 100.0 * correct / total
    return avg_loss, avg_cls, avg_aux, acc


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


# ==========================================================
# 3. Main
# ==========================================================

def main():
    print('=' * 80)
    print('LSK-FusionNet Training (MUUFL) - with aux_loss + Focal Loss')
    print(f'Config: gamma={FOCAL_GAMMA}, alpha={USE_FOCAL_ALPHA}')
    print('=' * 80)

    set_seed(RANDOM_SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_iter, test_iter, total_iter, all_iter, y, total_indices, all_indices, class_counts = create_data_loader()

    # class weights (仍然计算，供 focal alpha 或你之后做 weighted CE 用)
    counts = np.array(class_counts)
    weights = 1.0 / np.sqrt(counts)
    weights = weights / weights.sum() * len(counts)
    class_weights = torch.FloatTensor(weights).to(device)
    print(f"\nClass Weights: \n{weights}")

    # ======= 使用 Focal Loss =======
    alpha = class_weights if USE_FOCAL_ALPHA else None
    criterion = FocalLoss(gamma=FOCAL_GAMMA, alpha=alpha)

    model = model_pkg.TTTFusionNet(
        in_ch_hsi=PCA_COMPONENTS,
        in_ch_lidar=1,
        num_classes=NUM_CLASSES,
        emb_dim=EMB_DIM,
        depth=DEPTH
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    os.makedirs(SAVE_DIR_PARAMS, exist_ok=True)
    os.makedirs(SAVE_DIR_RESULTS, exist_ok=True)
    os.makedirs(SAVE_DIR_MAPS, exist_ok=True)

    best_acc = 0.0
    pbar = tqdm(range(EPOCHS), desc="Training")
    tic = time.time()

    for epoch in pbar:
        avg_loss, avg_cls, avg_aux, train_acc = train(
            model, train_iter, optimizer, scheduler, criterion, device
        )

        pbar.set_postfix({
            'Loss': f'{avg_loss:.4f}',
            'Cls': f'{avg_cls:.4f}',
            'Aux': f'{avg_aux:.4f}',
            'Tr_Acc': f'{train_acc:.2f}%',
            'Best': f'{best_acc:.2f}%'
        })

        if (epoch + 1) % 10 == 0 or (epoch + 1) == EPOCHS:
            y_pred, y_true = test(model, test_iter, device)
            test_acc = accuracy_score(y_true, y_pred) * 100.0

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), f'{SAVE_DIR_PARAMS}/LSKNet_MUUFL_best.pth')
                tqdm.write(f"Epoch {epoch + 1}: New Best Test Acc: {test_acc:.2f}%")

    toc = time.time()
    print(f'\nTraining Time: {toc - tic:.2f}s')
    print(f'Best Test Accuracy: {best_acc:.2f}%')

    print('\nLoading best model for evaluation...')
    model.load_state_dict(torch.load(f'{SAVE_DIR_PARAMS}/LSKNet_MUUFL_best.pth', map_location=device))
    y_pred, y_true = test(model, test_iter, device)

    target_names = [
        'Trees', 'Mostly Grass', 'Mixed Ground Surface', 'Dirt and Sand',
        'Road', 'Water', 'Buildings Shadow', 'Buildings',
        'Sidewalk', 'Yellow Curb', 'Cloth Panels'
    ]

    classification = classification_report(y_true, y_pred, digits=4, target_names=target_names)
    oa = accuracy_score(y_true, y_pred) * 100.0
    confusion = confusion_matrix(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred) * 100.0

    list_diag = np.diag(confusion)
    list_raw_sum = np.sum(confusion, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum)) * 100.0
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
        Utils.generate_png(
            total_iter, model, y, device, total_indices,
            f'{SAVE_DIR_MAPS}/LSKNet_MUUFL_labeled'
        )

    print('\nDone.')


if __name__ == '__main__':
    main()
