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
    """新增 highlight_points 参数用于指定需要突出的点"""
    if not record_list:
        print("No data to plot")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['b', 'g', 'r', 'm', 'y', 'k']
    linestyles = ['-', '--', '-.', ':']

    if not isinstance(record_list[0], list):
        record_list = [record_list]

    for i, series in enumerate(record_list):
        # 绘制主曲线
        ax.plot(x_ticks[:len(series)], series,
                label=line_names[i % len(line_names)],
                color=colors[i % len(colors)],
                linestyle=linestyles[i % len(linestyles)])

        # 添加红点标注（新增部分）
        if highlight_points:
            for point in highlight_points:
                if point[0] < len(series):  # 确保索引有效
                    ax.scatter(point[0], series[point[0]],  # 使用具体坐标
                               color='red',
                               zorder=5,  # 确保红点在曲线上方
                               s=20)  # 点的大小
                    # 可选添加文字标注
                    # ax.annotate(f'({point[0]}, {series[point[0]]:.2f})',
                    #             (point[0], series[point[0]]),
                    #             textcoords="offset points",
                    #             xytext=(10, -10),
                    #             ha='left',
                    #             arrowprops=dict(arrowstyle="->"))

    # 其余原有代码保持不变...
    ax.legend()
    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Test Accuracy')
    ax.grid(True)
    all_values = [value for series in record_list for value in series]  # 展平所有数据
    if not all_values:
        print("No valid data points")
        return

    y_min = min(all_values)  # 最小值
    y_max = max(all_values)  # 最大值

    # 设置纵坐标范围，根据数据动态调整
    ax.set_ylim(bottom=max(0, y_min - 0.05), top=min(1.0, y_max + 0.05))  # 上下留出5%的空白

    # 设置纵坐标刻度，每隔0.1显示一个刻度
    y_ticks = np.arange(np.floor(y_min * 10) / 10, np.ceil(y_max * 10) / 10 + 0.1, 0.1)
    ax.set_yticks(y_ticks)

    # 设置横坐标范围
    ax.set_xlim(left=0, right=len(x_ticks) - 1 if len(x_ticks) > 0 else 0)

    plt.tight_layout()  # 自动调整子图参数
    plt.show()


if __name__ == "__main__":
    file_path = './cnn_fedbuff_0.3b_40t.json'  # JSON文件路径
    line_names = ["FedBuff"]

    # 加载数据
    record_list = load_json_data(file_path)
    if not record_list:
        print("No data loaded. Exiting.")
        exit()

    # 准备x轴刻度
    x_ticks = list(range(len(record_list)))

    # 绘制数据
    plot_data([record_list], line_names, x_ticks)  # 注意将record_list包装在列表中


def find_highlight_points(record_list, threshold=0.1):
    """
    找出比前一个或后一个值小至少threshold的值
    返回格式：[(index, value), ...]
    """
    highlight_points = []

    for i in range(len(record_list)):
        current = record_list[i]
        has_condition = False

        # 检查前一个元素
        if i > 0:
            prev = record_list[i - 1]
            if prev - current >= threshold:
                has_condition = True

        # 检查后一个元素（如果前一个条件不满足）
        if not has_condition and i < len(record_list) - 1:
            next_val = record_list[i + 1]
            if next_val - current >= threshold:
                has_condition = True

        # 如果满足任一条件则记录
        if has_condition:
            highlight_points.append((i, round(current, 4)))  # 保留四位小数

    return highlight_points

if __name__ == "__main__":
    file_path = './fedbuff_own_0.7b_5t.json'
    line_names = ["FedBuff"]

    record_list = load_json_data(file_path)
    x_ticks = list(range(len(record_list)))

    # 指定需要突出的点（例如第50轮）
    highlight_points = find_highlight_points(record_list,0.015) # 直接使用坐标值
    # 然后修改标注部分：

    plot_data([record_list], line_names, x_ticks, highlight_points=highlight_points)