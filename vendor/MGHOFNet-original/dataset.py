import torch
import numpy as np
import torch.utils.data as Data
from sklearn import preprocessing
from tqdm import tqdm


class HSIDataset(Data.Dataset):
    def __init__(self, hsi_data, lidar_data, labels, indices, patch_size=11, return_label=True, cache_data=False,
                 augment=False):
        self.hsi_data = hsi_data
        self.lidar_data = lidar_data
        self.labels = labels
        self.indices = indices
        self.patch_len = (patch_size - 1) // 2
        self.return_label = return_label
        self.original_width = hsi_data.shape[1] - 2 * self.patch_len
        self.augment = augment  # 新增增强开关

        self.use_cache = cache_data
        self.cached_data = []

        if self.use_cache:
            print(f"Pre-loading {len(indices)} patches into memory (Augment={augment})...")
            for i in tqdm(range(len(indices)), desc="Caching Data", leave=False):
                self.cached_data.append(self._get_item_dynamic(i))

    def _get_item_dynamic(self, index):
        pixel_idx = self.indices[index]
        row = pixel_idx // self.original_width + self.patch_len
        col = pixel_idx % self.original_width + self.patch_len

        # HSI Patch
        hsi_patch = self.hsi_data[
                    row - self.patch_len: row + self.patch_len + 1,
                    col - self.patch_len: col + self.patch_len + 1,
                    :
                    ]
        # HSI: (H, W, C) -> (C, H, W)
        hsi_tensor = torch.from_numpy(hsi_patch.transpose(2, 0, 1)).float()

        # LiDAR Patch
        if self.lidar_data.ndim == 2:
            lidar_patch = self.lidar_data[
                          row - self.patch_len: row + self.patch_len + 1,
                          col - self.patch_len: col + self.patch_len + 1
                          ]
            lidar_tensor = torch.from_numpy(lidar_patch).float().unsqueeze(0)
        else:
            lidar_patch = self.lidar_data[
                          row - self.patch_len: row + self.patch_len + 1,
                          col - self.patch_len: col + self.patch_len + 1,
                          :
                          ]
            lidar_tensor = torch.from_numpy(lidar_patch.transpose(2, 0, 1)).float()

        label_val = 0
        if self.labels is not None:
            if self.labels.ndim == 2:
                label_val = self.labels.flatten()[pixel_idx]
            else:
                label_val = self.labels[pixel_idx]
            label_val = int(label_val) - 1

        return hsi_tensor, lidar_tensor, label_val

    def _augment_data(self, hsi, lidar):
        """简单的随机翻转和旋转"""
        # Random Horizontal Flip
        if torch.rand(1) < 0.5:
            hsi = torch.flip(hsi, [2])
            lidar = torch.flip(lidar, [2])

        # Random Vertical Flip
        if torch.rand(1) < 0.5:
            hsi = torch.flip(hsi, [1])
            lidar = torch.flip(lidar, [1])

        # Random Rotation (0, 90, 180, 270)
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            hsi = torch.rot90(hsi, k, [1, 2])
            lidar = torch.rot90(lidar, k, [1, 2])

        return hsi, lidar

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if self.use_cache:
            hsi, lidar, label = self.cached_data[index]
        else:
            if self.return_label:
                hsi, lidar, label = self._get_item_dynamic(index)
            else:
                hsi, lidar = self._get_item_dynamic(index)

        # 在获取数据时应用增强
        if self.augment:
            hsi, lidar = self._augment_data(hsi, lidar)

        if self.return_label:
            return hsi, lidar, label
        else:
            return hsi, lidar


# pad_and_normalize 保持不变
def pad_and_normalize(data, patch_size, mode='hsi'):
    pad_len = (patch_size - 1) // 2
    h, w = data.shape[:2]
    if mode == 'hsi':
        c = data.shape[2]
        data_flat = data.reshape(-1, c)
        data_flat = preprocessing.MinMaxScaler().fit_transform(data_flat)
        data_norm = data_flat.reshape(h, w, c)
        data_padded = np.pad(data_norm, ((pad_len, pad_len), (pad_len, pad_len), (0, 0)), 'constant')
    else:
        data_flat = data.reshape(-1, 1)
        data_flat = preprocessing.MinMaxScaler().fit_transform(data_flat)
        data_norm = data_flat.reshape(h, w)
        data_padded = np.pad(data_norm, ((pad_len, pad_len), (pad_len, pad_len)), 'constant')
    return data_padded


def get_dataloader(hsi_pad, lidar_pad, labels, indices, batch_size, patch_size, shuffle=False, return_label=True,
                   num_workers=4, cache_data=False, augment=False):
    dataset = HSIDataset(hsi_pad, lidar_pad, labels, indices, patch_size, return_label=return_label,
                         cache_data=cache_data, augment=augment)

    workers = 0 if cache_data else num_workers
    return Data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(workers > 0)
    )