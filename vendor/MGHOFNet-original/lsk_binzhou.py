import os
import time
from operator import truediv

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)
from tqdm import tqdm

import Utils
import dataset
import model as model_pkg


# ==========================================================
#                      Configuration
# ==========================================================
GENERATE_LABELED_MAP = True
GENERATE_ALL_MAP = False

EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
RANDOM_SEED = 42

PATCH_SIZE = 11
PCA_COMPONENTS = 30
TRAIN_SAMPLES_PER_CLASS = 50

SAVE_DIR_PARAMS = "cls_params/lsk_binzhou"
SAVE_DIR_RESULTS = "cls_result/lsk_binzhou"
SAVE_DIR_MAPS = "cls_map/lsk_binzhou"

NUM_CLASSES = 5
EMB_DIM = 128
DEPTH = 2


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_image_data(mat_path):
    data = sio.loadmat(mat_path)
    if "image_data" in data:
        return data["image_data"]
    valid_keys = [k for k in data.keys() if not k.startswith("__")]
    if len(valid_keys) == 1:
        return data[valid_keys[0]]
    raise KeyError(f"Cannot find image key in {mat_path}. Available keys: {valid_keys}")


def loadData():
    print("Loading Binzhou dataset...")
    base = "data/binzhou"

    hsi_path = os.path.join(base, "hsi.mat")
    lidar_path = os.path.join(base, "lidar.mat")
    label_path = os.path.join(base, "label.mat")

    if not (os.path.exists(hsi_path) and os.path.exists(lidar_path) and os.path.exists(label_path)):
        raise FileNotFoundError(
            f"Missing MAT files under {base}. Required: hsi.mat, lidar.mat, label.mat"
        )

    data_hsi = _load_image_data(hsi_path)
    data_lidar = _load_image_data(lidar_path)
    labels = _load_image_data(label_path)

    # Binzhou HSI is often saved as (C, H, W), convert to (H, W, C).
    if data_hsi.ndim == 3 and data_hsi.shape[0] < data_hsi.shape[1] and data_hsi.shape[0] < data_hsi.shape[2]:
        data_hsi = np.transpose(data_hsi, (1, 2, 0))
        print(f"Converted HSI from (C,H,W) to (H,W,C): {data_hsi.shape}")

    if labels.ndim == 3:
        labels = np.squeeze(labels)
    if data_lidar.ndim == 3:
        data_lidar = np.squeeze(data_lidar)

    return data_hsi, data_lidar, labels


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

    print(f"Selecting {num_train_per_class} samples per class for training...")

    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indices)

        nb_val = num_train_per_class
        if len(indices) < nb_val:
            print(f"Warning: Class {i + 1} only has {len(indices)} samples. Using 50% for train.")
            nb_val = max(1, int(len(indices) * 0.5))

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

    if X2.ndim == 2:
        X2 = X2[:, :, np.newaxis]

    print(f"HSI shape: {X1.shape}, LiDAR shape: {X2.shape}, Label shape: {y.shape}")
    print(f"Applying PCA (components={PCA_COMPONENTS})...")
    X1 = applyPCA(X1, numComponents=PCA_COMPONENTS)

    print(f"Padding and Normalizing (patch_size={PATCH_SIZE})...")
    X1_padded = dataset.pad_and_normalize(X1, PATCH_SIZE, mode="hsi")
    X2_padded = dataset.pad_and_normalize(X2, PATCH_SIZE, mode="lidar")

    gt = y.flatten().astype(int)

    train_indices, test_indices = select_traintest(gt, TRAIN_SAMPLES_PER_CLASS)

    total_indices = []
    m = int(max(gt))
    for i in range(m):
        total_indices += [j for j, x in enumerate(gt.ravel().tolist()) if x == i + 1]

    height, width = y.shape
    all_indices = np.arange(height * width)

    print(f"\nTrain samples: {len(train_indices)}")
    print(f"Test samples:  {len(test_indices)}")

    train_loader = dataset.get_dataloader(
        X1_padded,
        X2_padded,
        y,
        train_indices,
        batch_size=BATCH_SIZE,
        patch_size=PATCH_SIZE,
        shuffle=True,
        num_workers=0,
        cache_data=True,
        augment=False,
    )

    test_loader = dataset.get_dataloader(
        X1_padded,
        X2_padded,
        y,
        test_indices,
        batch_size=BATCH_SIZE,
        patch_size=PATCH_SIZE,
        shuffle=False,
        num_workers=4,
        cache_data=False,
        augment=False,
    )

    total_loader = None
    if GENERATE_LABELED_MAP:
        total_loader = dataset.get_dataloader(
            X1_padded,
            X2_padded,
            y,
            total_indices,
            BATCH_SIZE,
            PATCH_SIZE,
            shuffle=False,
            num_workers=4,
            cache_data=False,
        )

    all_loader = None
    if GENERATE_ALL_MAP:
        all_loader = dataset.get_dataloader(
            X1_padded,
            X2_padded,
            y,
            all_indices,
            BATCH_SIZE,
            PATCH_SIZE,
            shuffle=False,
            num_workers=4,
            cache_data=False,
        )

    return train_loader, test_loader, total_loader, all_loader, y, total_indices, all_indices


def train(model, train_loader, optimizer, scheduler, criterion, device):
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
    return total_loss / len(train_loader), 100.0 * correct / total


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
    print("=" * 80)
    print("LSK-FusionNet Training (Binzhou)")
    print(f"Config: EMB_DIM={EMB_DIM}, Samples/Class={TRAIN_SAMPLES_PER_CLASS}")
    print("=" * 80)

    set_seed(RANDOM_SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_iter, test_iter, total_iter, all_iter, y, total_indices, all_indices = create_data_loader()
    criterion = nn.CrossEntropyLoss()

    model = model_pkg.TTTFusionNet(
        in_ch_hsi=PCA_COMPONENTS,
        in_ch_lidar=1,
        num_classes=NUM_CLASSES,
        emb_dim=EMB_DIM,
        depth=DEPTH,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    os.makedirs(SAVE_DIR_PARAMS, exist_ok=True)
    os.makedirs(SAVE_DIR_RESULTS, exist_ok=True)
    os.makedirs(SAVE_DIR_MAPS, exist_ok=True)

    best_acc = 0
    pbar = tqdm(range(EPOCHS), desc="Training")
    tic = time.time()

    for epoch in pbar:
        loss, train_acc = train(model, train_iter, optimizer, scheduler, criterion, device)

        pbar.set_postfix({"Loss": f"{loss:.4f}", "Tr_Acc": f"{train_acc:.2f}%", "Best": f"{best_acc:.2f}%"})

        if (epoch + 1) % 10 == 0 or (epoch + 1) == EPOCHS:
            y_pred, y_true = test(model, test_iter, device)
            test_acc = accuracy_score(y_true, y_pred) * 100

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), f"{SAVE_DIR_PARAMS}/LSKNet_Binzhou_best.pth")
                tqdm.write(f"Epoch {epoch + 1}: New Best Test Acc: {test_acc:.2f}%")

    toc = time.time()
    print(f"\nTraining time: {(toc - tic):.2f}s")
    print(f"Best Test Accuracy: {best_acc:.2f}%")

    print("\nLoading best model for evaluation...")
    model.load_state_dict(torch.load(f"{SAVE_DIR_PARAMS}/LSKNet_Binzhou_best.pth"))
    y_pred, y_true = test(model, test_iter, device)

    target_names = [f"Class {i + 1}" for i in range(NUM_CLASSES)]
    classification = classification_report(y_true, y_pred, digits=4, target_names=target_names)
    oa = accuracy_score(y_true, y_pred) * 100
    confusion = confusion_matrix(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred) * 100

    list_diag = np.diag(confusion)
    list_raw_sum = np.sum(confusion, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum)) * 100
    aa = np.mean(each_acc)

    print("\nClassification Report:")
    print(classification)
    print(f"OA: {oa:.2f}% | AA: {aa:.2f}% | Kappa: {kappa:.2f}%")

    with open(f"{SAVE_DIR_RESULTS}/LSKNet_Binzhou_report.txt", "w", encoding="utf-8") as f:
        f.write("Binzhou Dataset (5 Classes)\n")
        f.write(f"Train Samples/Class: {TRAIN_SAMPLES_PER_CLASS}\n")
        f.write(f"Patch Size: {PATCH_SIZE}, PCA Components: {PCA_COMPONENTS}\n\n")
        f.write(f"OA: {oa:.2f}%\nAA: {aa:.2f}%\nKappa: {kappa:.2f}%\n\n")
        f.write(classification)
        f.write(f"\n\nPer Class Acc:\n{each_acc}")
        f.write(f"\n\nConfusion Matrix:\n{confusion}")

    if GENERATE_LABELED_MAP and total_iter is not None:
        print("\nGenerating labeled classification map...")
        Utils.generate_png(total_iter, model, y, device, total_indices, f"{SAVE_DIR_MAPS}/LSKNet_Binzhou_labeled")

    if GENERATE_ALL_MAP and all_iter is not None:
        print("\nGenerating full classification map...")
        Utils.generate_all_png(all_iter, model, y, device, all_indices, f"{SAVE_DIR_MAPS}/LSKNet_Binzhou_all")

    print("\nDone.")


if __name__ == "__main__":
    main()
