import os
import json
import matplotlib.pyplot as plt
import numpy as np

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
                    record_list.append(data['test_accuracy'])
                else:
                    print(f"'test_accuracy' not found in {file_path}")
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error reading {file_path}: {e}")
    return record_list, filenames

def plot_data(record_list, line_names, title=None):
    """
    绘制折线图，可选添加标题在图像正上方（使用Times New Roman字体）。
    """
    if len(record_list) != len(line_names):
        raise ValueError("The length of record_list and line_names must be the same.")

    fig, ax = plt.subplots(figsize=(8, 6))

    if title:
        plt.suptitle(title, fontsize=14, fontname='Times New Roman')

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

    # plt.tight_layout(rect=[0, 0, 1, 0.95])  # 给 suptitle 留出空间

    plt.savefig("./photo/acc_FedBuff_plot", dpi=600, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    folder_path = './acc'  # 文件夹路径

    # 指定要读取的文件名（不含扩展名），顺序自定义
    custom_filenames = ["FedBuff_0.3b_10t", "FedBuff_0.3b_100t", "FedBuff_0.7b_10t", "FedBuff_0.7b_100t"]

    # 加载数据
    record_list, line_names = load_json_data_by_filenames(custom_filenames, folder_path)

    # 绘制图表
    plot_data(record_list, line_names, title="The accuracy of the FedBuff algorithm on the Fashion-MNIST dataset")