import os
import json
import numpy as np
import matplotlib.pyplot as plt
def load_json_data_by_filenames(filenames, folder_path):
    """
    按指定顺序加载给定文件名列表中的JSON文件，提取'test_accuracy'字段。
    """
    record_list = []
    for name in filenames:
        file_path = os.path.join(folder_path, f"{name}.json")
        try:
            with open(file_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
                if 'test_accuracy' in data:
                    record_list.append(data['test_accuracy'][:601])
                else:
                    print(f"'test_accuracy' not found in {file_path}")
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error reading {file_path}: {e}")
    return record_list, filenames

def plot_data(record_list, line_names, title=None):
    plt.rcParams.update({'font.size': 14})  # 全局字体大小设置
    """
    绘制折线图，可选添加标题在图像正上方（使用Times New Roman字体）。
    """
    if len(record_list) != len(line_names):
        raise ValueError("The length of record_list and line_names must be the same.")

    # 增大图形高度并调整子图位置（关键修改点1）
    fig, ax = plt.subplots(figsize=(8, 6.5))  # 高度从6增加到6.5
    plt.subplots_adjust(bottom=0.15)  # 关键修改点2：给底部标题留空间

    if title:
        # 关键修改点3：调整y值和垂直对齐方式
        plt.suptitle(title, fontsize=14, fontname='SimSun',
                    y=0.02, va='bottom')  # y更小，va指定对齐

    colors = ['b', 'g', 'r', 'm', 'y', 'k']
    linestyles = ['-.']

    for i, series in enumerate(record_list):
        x = list(range(len(series)))
        ax.plot(x, series, label=line_names[i],
                color=colors[i % len(colors)],
                linestyle=linestyles[i % len(linestyles)])

    ax.legend()
    ax.set_xlabel('Communication Round', fontname='Times New Roman')
    ax.set_ylabel('Test Accuracy', fontname='Times New Roman')
    ax.grid(True)

    all_values = [val for series in record_list for val in series]
    y_min, y_max = min(all_values), max(all_values)
    ax.set_ylim(bottom=max(0, y_min - 0.05), top=y_max + 0.05)
    y_ticks = np.arange(np.floor(y_min * 10) / 10, np.ceil(y_max * 10) / 10 + 0.1, 0.1)
    ax.set_yticks(y_ticks)

    max_len = max(len(series) for series in record_list)
    ax.set_xlim(left=0, right=max_len - 1)

    # 移除tight_layout（关键修改点4）
    plt.savefig("./photo/acc_3b40t_plot.png", bbox_inches='tight', dpi=300)  # 确保保存为PNG
    plt.show()


if __name__ == "__main__":
    folder_path = './'  # 文件夹路径

    # 指定要读取的文件名（不含扩展名），顺序自定义
    custom_filenames = ["FedAsync_0.3b_40t","KAFL_0.3b_40t", "ca2fl_0.3b_40t", "FedBuff_0.3b_40t", "ours_0.3b_40t"]

    # 加载数据
    record_list, line_names = load_json_data_by_filenames(custom_filenames, folder_path)

    # 绘制图表
    plot_data(record_list, line_names, title="(b) 0.3b40t客户端异构设置")