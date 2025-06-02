import os
import json
import matplotlib.pyplot as plt
import numpy as np

import os
import json
import matplotlib.pyplot as plt
import numpy as np


def load_json_data(file_path):
    """从指定JSON文件加载数据并提取test_accuracy字段"""
    try:
        with open(file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
            if 'test_accuracy' in data:
                return data['test_accuracy']
            else:
                print(f"Warning: 'test_accuracy' not found in {file_path}")
                return []
    except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
        print(f"Error reading {file_path}: {e}")
        return []


def plot_data(record_list, line_names, x_ticks, highlight_points=None):
    """支持多组数据和对应的突出点"""
    if not record_list or all(not sublist for sublist in record_list):
        print("No data to plot")
        return

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    colors = ['b', 'g', 'r', 'm', 'y', 'k']
    linestyles = ['-']

    for i, series in enumerate(record_list):
        if not series:  # 跳过空数据
            continue

        # 绘制主曲线
        ax.plot(x_ticks[:len(series)], series,
                label=line_names[i % len(line_names)],
                color=colors[i % len(colors)],
                linestyle=linestyles[i % len(linestyles)],
                zorder=3)

        # 添加红点标注
        if highlight_points and i < len(highlight_points):
            for point in highlight_points[i]:
                if point[0] < len(series):
                    ax.scatter(point[0], series[point[0]],
                               color='red', zorder=5, s=30,
                               edgecolors='black', linewidths=0.5)

    ax.legend()
    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Test Accuracy')
    ax.grid(True, zorder=1)

    # 动态设置坐标轴范围
    all_values = [v for series in record_list if series for v in series]
    if not all_values:
        print("No valid data points")
        return

    y_min = min(all_values)
    y_max = max(all_values)
    ax.set_ylim(max(0, y_min - 0.05), min(1.0, y_max + 0.05))
    ax.set_yticks(np.arange(np.floor(y_min * 10) / 10, np.ceil(y_max * 10) / 10 + 0.1, 0.1))

    max_x = max(len(series) for series in record_list if series)
    ax.set_xlim(0, max_x - 1 if max_x > 0 else 0)
    ax.set_xticks(np.arange(0, max_x, max(1, max_x // 10)))

    plt.savefig('comparison_plot.png', dpi=600, bbox_inches='tight')
    plt.show()

def find_highlight_points(record_list, threshold=0.3):
    """为单个数据集找突出点"""
    points = []
    for i in range(1, len(record_list)):
        if record_list[i - 1] - record_list[i] >= threshold:
            points.append((i, round(record_list[i], 4)))
    return points
def smooth_data(series):
    """每四个相邻点替换为它们的均值，保持数据长度不变"""
    smoothed = []
    for i in range(0, len(series), 4):  # 步长改为4
        group = series[i:i+4]          # 获取连续4个点
        if group:  # 确保非空组
            avg = sum(group) / len(group)
            # 用平均值填充原始数据长度（最多4个）
            smoothed.extend([avg] * len(group))
    return smoothed

def average_every_two(data):
    """（保留函数但修改实现）每两个点合并为一个均值点，返回新数据和对应轮次"""
    averaged_data = []
    x_ticks = []
    for i in range(0, len(data), 2):
        group = data[i:i+2]
        if group:
            avg = np.mean(group)
            averaged_data.append(avg)
            x_ticks.append(i)  # 记录该组起始轮次
    return averaged_data, x_ticks

if __name__ == "__main__":
    # 配置两个文件和对应的曲线名称
    file_paths = [
        './fedbuff_own_0.7b_5t.json',
        './fedbuff_own_0.7b_120t.json'  # 请替换为实际路径
    ]
    line_names = ["FedBuff v1", "FedBuff v2"]

    # 加载数据
    datasets = [load_json_data(fp) for fp in file_paths]
    datasets[0]=smooth_data(datasets[0])

    # 计算x轴刻度（取最大长度）
    max_length = max(len(d) for d in datasets if d)
    x_ticks = list(range(max_length))

    # 计算各组的突出点
    # highlight_points = [find_highlight_points(d, 0.01) for d in datasets]
    highlight_points=[]
    # 绘制对比图
    plot_data(
        record_list=datasets,
        line_names=line_names,
        x_ticks=x_ticks,
        highlight_points=highlight_points
    )