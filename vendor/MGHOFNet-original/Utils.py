import numpy as np
import matplotlib.pyplot as plt
import torch


def set_figsize(figsize=(3.5, 2.5)):
    """设置 matplotlib 图表大小"""
    plt.rcParams['figure.figsize'] = figsize


def classification_map(map_data, ground_truth, dpi, save_path):
    """
    绘制并保存分类图
    Args:
        map_data: RGB 图像数据
        ground_truth: 原始 Ground Truth (用于确定尺寸)
        dpi: 图像分辨率
        save_path: 保存路径
    """
    fig = plt.figure(frameon=False)
    # 根据 GT 尺寸设置图片大小
    fig.set_size_inches(ground_truth.shape[1] * 2.0 / dpi,
                        ground_truth.shape[0] * 2.0 / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)
    ax.imshow(map_data)
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)  # 关闭图形释放内存


def list_to_colormap(x_list):
    """
    将标签索引映射到 RGB 颜色
    """
    # 颜色表定义 (0-18)
    # 注意：这里直接对应原始代码中的颜色值
    colors = [
        [58, 138, 71],  # 0
        [204, 180, 206],  # 1
        [150, 84, 54],  # 2
        [251, 193, 150],  # 3
        [137, 145, 200],  # 4
        [238, 45, 42],  # 5
        [86, 132, 193],  # 6
        [128, 128, 128],  # 7
        [128, 0, 0],  # 8
        [128, 128, 0],  # 9
        [0, 128, 0],  # 10
        [128, 0, 128],  # 11
        [0, 128, 128],  # 12
        [0, 0, 128],  # 13
        [255, 165, 0],  # 14
        [255, 215, 0],  # 15
        [0, 0, 0],  # 16 (Background in Prediction)
        [215, 255, 0],  # 17 (Background in GT)
        [0, 255, 215]  # 18
    ]

    # 转换为 0-1 范围的 float
    colors = np.array(colors) / 255.

    y = np.zeros((x_list.shape[0], 3))

    for index, item in enumerate(x_list):
        item = int(item)
        if item == -1:
            # 特殊处理 -1 (通常未定义或背景)，对应蓝色
            y[index] = np.array([0, 0, 255]) / 255.
        elif 0 <= item < len(colors):
            y[index] = colors[item]
        else:
            # 默认黑色防止越界
            y[index] = np.array([0, 0, 0])

    return y


def generate_png(data_loader, net, gt_hsi, device, indices, path):
    """
    生成预测分类图
    Args:
        data_loader: 数据加载器 (Dataset应返回 hsi, lidar, [label])
        net: 训练好的模型
        gt_hsi: 原始 HSI 数据的形状参考 (H, W) 或 GT
        device: 计算设备
        indices: 预测像素在全图中的一维索引
        path: 保存路径前缀
    """
    print(f'Start generating map for {len(indices)} pixels...')
    pred_test = []
    net.eval()

    with torch.no_grad():
        for batch in data_loader:
            # 兼容 dataset 返回 (hsi, lidar, label) 或 (hsi, lidar)
            X1 = batch[0].to(device)
            X2 = batch[1].to(device)

            # 模型推理
            outputs = net(X1, X2)
            pred_test.extend(outputs.cpu().argmax(axis=1).detach().numpy())

    # 处理 Ground Truth 和 预测图背景
    gt = gt_hsi.flatten()
    x_label = np.zeros(gt.shape)

    # 原始逻辑：将背景(0)在GT显示为17号色，在预测图中显示为16号色(黑色)
    # 并将标签整体 -1 以适配网络输出 (假设网络输出 0~N-1)
    for i in range(len(gt)):
        if gt[i] <= 0:
            gt[i] = 17
            x_label[i] = 16

    gt = gt[:] - 1

    # 将预测结果填入对应位置
    # 注意：indices 必须是 int 类型数组
    if len(indices) > 0:
        x_label[indices] = pred_test

    x = np.ravel(x_label)

    # 颜色映射
    y_list = list_to_colormap(x)
    y_gt = list_to_colormap(gt)

    # 重塑为图像尺寸
    h, w = gt_hsi.shape[0], gt_hsi.shape[1]
    y_re = np.reshape(y_list, (h, w, 3))
    gt_re = np.reshape(y_gt, (h, w, 3))

    # 保存图片
    # classification_map(y_re, gt_hsi, 300, path + '.eps') # 如需EPS格式可取消注释
    classification_map(y_re, gt_hsi, 300, path + '.png')
    classification_map(gt_re, gt_hsi, 300, path + '_gt.png')

    print(f'Map saved to {path}.png')


def generate_all_png(data_loader, net, gt_hsi, device, indices, path):
    """
    全图生成接口，逻辑与 generate_png 相同，单独列出以便区分调用语义
    """
    print('Generating full classification map...')
    generate_png(data_loader, net, gt_hsi, device, indices, path)