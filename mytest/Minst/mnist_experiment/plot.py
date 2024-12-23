import os
import json
import matplotlib.pyplot as plt

folder_path = 'Record'  # 文件夹路径

# 获取文件夹中所有文件的列表
file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
record_list = []
# 循环遍历所有文件
for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)
    if file_name.endswith('.json'):  # 确保文件是JSON文件
        with open(file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
            # 检查固定字段是否存在并获取其值
            # if 'val_accuracy' in data:
            if 'test_loss' in data:
                fixed_field_value = data['test_loss']
                # fixed_field_value = data['val_accuracy']
                record_list.append(fixed_field_value)

# 假设data是你的包含8个列表的列表
import matplotlib.pyplot as plt

line_names = [
    "fedasync", "fedbalance"
]

# 创建一个新的图形对象和一个轴对象
fig, ax = plt.subplots(figsize=(8, 6))

# 在同一个轴对象上绘制每个子列表的折线图
for i, series in enumerate(record_list):
    ax.plot(series, label=line_names[i],linestyle='solid')

# 添加图例
ax.legend()

# 添加标题和轴标签
ax.set_xlabel('Communication Round')
ax.set_ylabel('Test_Accuracy')
# ax.set_ylim(0, 1)
ax.set_yticks([0,0.1, 0.2,0.3, 0.4,0.5, 0.6, 0.7,0.8,0.9, 1])
# 显示图形
plt.show()
