import os
import json
import matplotlib.pyplot as plt
import numpy as np

def load_json_data(folder_path,file_name):
    """从指定文件夹加载JSON文件并提取test_accuracy字段"""
    record_list = []
    for name in file_name:
        file_path = os.path.join(folder_path,  f"{name}.json")
        try:
            with open(file_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
                if 'lambda' in data:
                    record_list.append(data['lambda'][:3001])
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error reading {file_name}: {e}")
    return record_list

def average_every_100(data):
    """每100个数据点求平均值，并将平均值作为该组的起始点的值"""
    averaged_data = []
    x_ticks = []  # 用于存储横坐标的真实值
    for series in data:
        # 将数据分成每100个一组
        groups = [series[i:i + 10] for i in range(0, len(series), 10)]
        # 对每组求平均值
        averaged_series = [np.mean(group) for group in groups]
        averaged_data.append(averaged_series)
        # 生成对应的横坐标（每组的起始点的索引）
        x_ticks = [i * 10 for i in range(len(groups))]  # 每组的起始round数
    return averaged_data, x_ticks

def plot_data(record_list, line_names, x_ticks,title):
    """绘制折线图"""
    if len(record_list) != len(line_names):
        raise ValueError("The length of record_list and line_names must be the same.")

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.15)  # 调整底部边距
    colors = ['b', 'g', 'r',  'm', 'y', 'k']  # 定义颜色列表
    linestyles = ['-.']  # 定义线型列表
    if title:
        plt.suptitle(title, fontsize=14, fontname='SimSun',
                    y=0.02, va='bottom')  # y更小，va指定对齐
    for i, series in enumerate(record_list):
        ax.plot(x_ticks, series, label=line_names[i], color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)])

    ax.legend()
    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Test_Accuracy')
    ax.grid(True)  # 添加网格线

    # 动态计算纵坐标范围
    all_values = [value for series in record_list for value in series]  # 展平所有数据
    y_min = min(all_values)  # 最小值
    y_max = max(all_values)  # 最大值

    # 设置纵坐标范围，根据数据动态调整
    # ax.set_ylim(bottom=max(0, y_min - 0.05), top=y_max + 0.05)  # 上下留出5%的空白

    # 设置纵坐标刻度，每隔0.1显示一个刻度
    # y_ticks = np.arange(np.floor(y_min * 10) / 10, np.ceil(y_max * 10) / 10 + 0.1, 0.1)

    # ax.set_yticks(y_ticks)

    # 设置横坐标范围，去掉多余的空白位置
    ax.set_xlim(left=0, right=x_ticks[-1])  # 横坐标从0开始，到最后一个数据点结束
    plt.savefig("./photo/lambda_plot",  bbox_inches='tight',dpi=300)
    plt.show()

if __name__ == "__main__":
    folder_path = './'  # 文件夹路径
    line_names = ["FedAsync_0.7b_100t","KAFL_0.7b_100t", "ca2fl_0.7b_100t", "FedBuff_0.7b_100t", "ours_0.7b_100t"]

    # 加载数据
    record_list = load_json_data(folder_path,line_names)
    # 每100个点求平均值，并获取对应的横坐标
    averaged_record_list, x_ticks = average_every_100(record_list)
    # 绘制图表
    plot_data(averaged_record_list, line_names, x_ticks,title="(f) 0.7b100t客户端异构设置")