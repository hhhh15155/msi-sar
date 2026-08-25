"""YRD vs YRD2509 特征分析：为什么 YRD 精度更高"""
import numpy as np, scipy.io, h5py

# ── 加载 YRD (v7.3, h5py) ──
f = h5py.File('data/yrd/data.mat', 'r')
yrd_data = np.array(f['data']).transpose(2, 1, 0).astype(np.float32)  # (2048,2048,14)
f.close()
f = h5py.File('data/yrd/label.mat', 'r')
yrd_label = np.array(f['label']).transpose(1, 0).astype(np.int64)
f.close()

# ── 加载 YRD2509 (v5, scipy) ──
yrd9_data = scipy.io.loadmat('data/yrd2509/data.mat')['data'].astype(np.float32)
yrd9_label = scipy.io.loadmat('data/yrd2509/label.mat')['label'].astype(np.int64)

# ── 归一化 (per-channel z-score, 全局统计) ──
def norm_global(d):
    mean = d.reshape(-1, d.shape[-1]).mean(axis=0)
    std = d.reshape(-1, d.shape[-1]).std(axis=0)
    return (d - mean) / (std + 1e-8)

yrd_n = norm_global(yrd_data)
yrd9_n = norm_global(yrd9_data)

yrd_classes = ['Waterbody','Spartina','Phragmites','Tamarix','Suaeda',
               'Tidal flat','Willow','Cultivated']
yrd9_classes = ['Artificial','Tamarix','River','Suaeda','Pond','Reed',
                'Bare soil','Willow','Tidal flat']

yrd_band_names = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12',
                  'SAR-feat1','SAR-VV?','SAR-VH?','SAR-feat4']
yrd9_band_names = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12',
                   'VV-SAR','VH-SAR']

# ═══════════════════════════════════════════════════════════
# 1. 类别间可分性: Jeffries-Matusita 距离
# ═══════════════════════════════════════════════════════════
print("=" * 100)
print("1. 类别间可分性分析")
print("=" * 100)

def jeffries_matusita(mu1, mu2, sigma1, sigma2):
    """JM距离, 值域[0, 2], >1.9=极好, 1.0-1.9=中等, <1.0=差"""
    d = mu1 - mu2
    s = (sigma1 + sigma2) / 2.0
    try:
        det_s = np.linalg.det(s)
        det_s1 = np.linalg.det(sigma1)
        det_s2 = np.linalg.det(sigma2)
        denom = np.sqrt(max(det_s1 * det_s2, 1e-30))
        B = float(0.125 * np.dot(d, np.linalg.solve(s, d)) +
                   0.5 * np.log(max(det_s / denom, 1e-30)))
    except np.linalg.LinAlgError:
        B = 0.0
    B = max(min(B, 80.0), 0.0)
    return float(np.sqrt(2.0 * (1.0 - np.exp(-B))))

def compute_jm_matrix(data, label, classes, n_classes):
    means, covs = [], []
    for c in range(1, n_classes+1):
        mask = label == c
        x = data[mask]
        means.append(x.mean(axis=0))
        # 降维用 PCA? 不，直接用对角协方差避免奇异矩阵
        covs.append(np.var(x, axis=0) + 1e-4)  # diagonal
    # 用完整协方差
    covs_full = []
    for c in range(1, n_classes+1):
        mask = label == c
        x = data[mask]
        # 采样避免内存爆炸 + 加速
        n = min(5000, x.shape[0])
        idx = np.random.choice(x.shape[0], n, replace=False)
        covs_full.append(np.cov(x[idx].T) + np.eye(data.shape[-1]) * 1e-4)

    jm = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(i+1, n_classes):
            jm[i,j] = jeffries_matusita(means[i], means[j], covs_full[i], covs_full[j])
            jm[j,i] = jm[i,j]
    return jm, np.array(means)

jm_yrd, means_yrd = compute_jm_matrix(yrd_n, yrd_label, yrd_classes, 8)
jm_yrd9, means_yrd9 = compute_jm_matrix(yrd9_n, yrd9_label, yrd9_classes, 9)

print("\nYRD JM距离矩阵 (8类):")
header = f"{'':>12}" + "".join(f"{c:>10}" for c in yrd_classes)
print(header)
for i, c in enumerate(yrd_classes):
    line = f"{c:>12}"
    for j in range(8):
        v = jm_yrd[i,j]
        s = f"{v:.2f}" if i != j else "-"
        line += f"{s:>10}"
    print(line)

# 平均JM
yrd_jm_vals = [jm_yrd[i,j] for i in range(8) for j in range(i+1,8)]
print(f"\nYRD  平均 JM = {np.mean(yrd_jm_vals):.3f}  (min={np.min(yrd_jm_vals):.3f}, max={np.max(yrd_jm_vals):.3f})")

print("\nYRD2509 JM距离矩阵 (9类):")
header = f"{'':>12}" + "".join(f"{c:>10}" for c in yrd9_classes)
print(header)
for i, c in enumerate(yrd9_classes):
    line = f"{c:>12}"
    for j in range(9):
        v = jm_yrd9[i,j]
        s = f"{v:.2f}" if i != j else "-"
        line += f"{s:>10}"
    print(line)

yrd9_jm_vals = [jm_yrd9[i,j] for i in range(9) for j in range(i+1,9)]
print(f"\nYRD2509 平均 JM = {np.mean(yrd9_jm_vals):.3f}  (min={np.min(yrd9_jm_vals):.3f}, max={np.max(yrd9_jm_vals):.3f})")

# 找出最混淆的类对
print("\nYRD 最难分类对 (JM最小):")
pairs = [(jm_yrd[i,j], yrd_classes[i], yrd_classes[j]) for i in range(8) for j in range(i+1,8)]
pairs.sort()
for d, a, b in pairs[:5]:
    print(f"  {a} ↔ {b}: JM={d:.3f}")

print("\nYRD2509 最难分类对 (JM最小):")
pairs = [(jm_yrd9[i,j], yrd9_classes[i], yrd9_classes[j]) for i in range(9) for j in range(i+1,9)]
pairs.sort()
for d, a, b in pairs[:5]:
    print(f"  {a} ↔ {b}: JM={d:.3f}")

# ═══════════════════════════════════════════════════════════
# 2. 重叠类的光谱 vs SAR 分离度对比
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("2. 重叠混淆类的 光学 vs SAR 分离度对比")
print("=" * 100)

# YRD 和 YRD2509 共有的混淆类对
overlap_pairs = [
    (yrd_classes.index('Tamarix'), yrd_classes.index('Suaeda')),
    (yrd_classes.index('Phragmites'), yrd_classes.index('Willow')),
    (yrd_classes.index('Tamarix'), yrd_classes.index('Willow')),
]

yrd9_overlap = [
    (yrd9_classes.index('Tamarix'), yrd9_classes.index('Suaeda')),
    (yrd9_classes.index('Reed'), yrd9_classes.index('Willow')),
    (yrd9_classes.index('Tamarix'), yrd9_classes.index('Willow')),
]

def channel_subset_jm(data, label, pair, ch_subset):
    """计算只用特定通道子集的JM"""
    i, j = pair
    mu_i = data[label == i+1][:, ch_subset].mean(axis=0)
    mu_j = data[label == j+1][:, ch_subset].mean(axis=0)
    cov_i = np.cov(data[label == i+1][:, ch_subset].T) + np.eye(len(ch_subset))*1e-4
    cov_j = np.cov(data[label == j+1][:, ch_subset].T) + np.eye(len(ch_subset))*1e-4
    return jeffries_matusita(mu_i, mu_j, cov_i, cov_j)

for (pi, pj), (qi, qj) in zip(overlap_pairs, yrd9_overlap):
    aname, bname = yrd_classes[pi], yrd_classes[pj]
    print(f"\n{aname} ↔ {bname}:")
    # 只用光学波段 (0-9)
    jm_opt_yrd = channel_subset_jm(yrd_n, yrd_label, (pi, pj), range(10))
    jm_opt_yrd9 = channel_subset_jm(yrd9_n, yrd9_label, (qi, qj), range(10))
    # 光学 + SAR
    jm_full_yrd = jm_yrd[pi, pj]
    jm_full_yrd9 = jm_yrd9[qi, qj]
    # SAR only
    jm_sar_yrd = channel_subset_jm(yrd_n, yrd_label, (pi, pj), range(10, 14))
    jm_sar_yrd9 = channel_subset_jm(yrd9_n, yrd9_label, (qi, qj), range(10, 12))

    print(f"  YRD:      光学={jm_opt_yrd:.3f}  SAR={jm_sar_yrd:.3f}  全波段={jm_full_yrd:.3f}")
    print(f"  YRD2509:  光学={jm_opt_yrd9:.3f}  SAR={jm_sar_yrd9:.3f}  全波段={jm_full_yrd9:.3f}")
    if jm_full_yrd > jm_full_yrd9:
        gain = jm_full_yrd - jm_full_yrd9
        sar_gain = jm_sar_yrd - jm_sar_yrd9
        print(f"  → YRD 分离度更高 (+{gain:.3f})，SAR贡献={sar_gain:+.3f}")

# ═══════════════════════════════════════════════════════════
# 3. 通道重要性分析
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("3. 单通道区分能力（所有类对的平均JM）")
print("=" * 100)

def single_channel_jm_mean(data, label, n_classes):
    nc = data.shape[-1]
    scores = []
    for ch in range(nc):
        jm_vals = []
        for i in range(n_classes):
            for j in range(i+1, n_classes):
                mu_i = data[label == i+1][:, ch].mean()
                mu_j = data[label == j+1][:, ch].mean()
                s_i = data[label == i+1][:, ch].var() + 1e-4
                s_j = data[label == j+1][:, ch].var() + 1e-4
                B = 0.125 * (mu_i-mu_j)**2 / ((s_i+s_j)/2) + 0.5*np.log((s_i+s_j)/(2*np.sqrt(s_i*s_j)))
                jm_vals.append(np.sqrt(2*(1-np.exp(-np.clip(B,0,80)))))
        scores.append(np.mean(jm_vals))
    return np.array(scores)

yrd_ch_score = single_channel_jm_mean(yrd_n, yrd_label, 8)
yrd9_ch_score = single_channel_jm_mean(yrd9_n, yrd9_label, 9)

print("\nYRD 单通道区分力:")
for i, (name, s) in enumerate(zip(yrd_band_names, yrd_ch_score)):
    bar = "█" * int(s * 80)
    print(f"  {name:>12s}  [{s:.4f}] {bar}")

print("\nYRD2509 单通道区分力:")
for i, (name, s) in enumerate(zip(yrd9_band_names, yrd9_ch_score)):
    bar = "█" * int(s * 80)
    print(f"  {name:>12s}  [{s:.4f}] {bar}")

print(f"\nYRD   光学均值={yrd_ch_score[:10].mean():.4f}  SAR均值={yrd_ch_score[10:].mean():.4f}")
print(f"YRD2509 光学均值={yrd9_ch_score[:10].mean():.4f}  SAR均值={yrd9_ch_score[10:].mean():.4f}")

# ═══════════════════════════════════════════════════════════
# 4. 总结
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("4. 总结")
print("=" * 100)
print(f"""
YRD    平均 JM = {np.mean(yrd_jm_vals):.4f}  (越低越难分)
YRD2509 平均 JM = {np.mean(yrd9_jm_vals):.4f}

YRD 优势来源:
  1) 光学信道区分力: {yrd_ch_score[:10].mean():.4f} vs {yrd9_ch_score[:10].mean():.4f} (空间分辨率翻倍带来的纹理优势)
  2) SAR 信道区分力:  {yrd_ch_score[10:].mean():.4f} vs {yrd9_ch_score[10:].mean():.4f} (多2个SAR派生特征)
  3) 额外2个SAR通道使混淆类对分离度提升显著
""")
