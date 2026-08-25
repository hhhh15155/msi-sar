import os, torch, itertools, time
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from tqdm import tqdm
import model as model_pkg
import dataset, Utils

# --- 配置与原始脚本对齐 ---
SEARCH_EPOCHS = 200
BATCH_SIZE = 64
RANDOM_SEED = 42
PATCH_SIZE = 11
PCA_COMPONENTS = 30
TRAIN_SAMPLES_PER_CLASS = 20
NUM_CLASSES = 11
SEARCH_PARAMS = {'depth': [1, 2, 3], 'emb_dim': [64, 128, 256]}
RESULTS_ROOT = 'cls_result/lsk_muufl_param_search'


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_experiment():
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(RANDOM_SEED)

    # 1. 加载数据与索引 (直接复用 lsk_muufl.py 的逻辑以确保完全对齐)
    # 确保 lsk_muufl.py 在同一目录下
    from lsk_muufl import create_data_loader
    train_iter, test_iter, _, _, y_gt, total_indices, all_indices, _ = create_data_loader()

    combinations = [dict(zip(SEARCH_PARAMS.keys(), v)) for v in itertools.product(*SEARCH_PARAMS.values())]
    results = []

    for idx, config in enumerate(combinations):
        d, e = config['depth'], config['emb_dim']
        print(f"\n[{idx + 1}/{len(combinations)}] Config: Depth={d}, EmbDim={e}")

        model = model_pkg.TTTFusionNet(PCA_COMPONENTS, 1, NUM_CLASSES, e, d).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SEARCH_EPOCHS)

        # --- 核心配置: 无权重 Loss ---
        criterion = nn.CrossEntropyLoss()

        best_oa = 0
        for epoch in range(SEARCH_EPOCHS):
            model.train()
            for hsi, lidar, labels in train_iter:
                hsi, lidar, labels = hsi.to(device), lidar.to(device), labels.to(device)

                # === 修正部分：标准的 PyTorch 训练步骤 ===
                optimizer.zero_grad()  # 1. 清空梯度
                outputs = model(hsi, lidar)  # 2. 前向传播
                loss = criterion(outputs, labels)  # 3. 计算 Loss
                loss.backward()  # 4. 反向传播
                optimizer.step()  # 5. 更新参数
                # =======================================

            scheduler.step()

            # 每 10 轮测试一次
            if (epoch + 1) % 10 == 0 or (epoch + 1) == SEARCH_EPOCHS:
                model.eval()
                y_p, y_t = [], []
                with torch.no_grad():
                    for hsi, lidar, labels in test_iter:
                        # 获取预测类别索引
                        preds = model(hsi.to(device), lidar.to(device)).argmax(1).cpu().numpy()
                        y_p.extend(preds)
                        y_t.extend(labels.numpy())

                oa = accuracy_score(y_t, y_p) * 100
                if oa > best_oa:
                    best_oa = oa

                # === 新增：打印进度 ===
                print(f"  Epoch {epoch + 1:03d}: OA = {oa:.2f}% (Best = {best_oa:.2f}%)")

        results.append({'depth': d, 'emb_dim': e, 'OA': best_oa})

        # 实时保存结果，防止中断丢失
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(RESULTS_ROOT, 'summary_param.csv'), index=False)
        print(f"Finished config. Result saved.")

    print(f"\nAll experiments done! Results saved to {RESULTS_ROOT}/summary_param.csv")


if __name__ == '__main__':
    run_experiment()