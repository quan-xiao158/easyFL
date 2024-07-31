# import flgo.experiment.analyzer as al
# task1 = './plotdir' # task name
# task2 = './dir0.3_mnist_straless0' # task name
# analysis_plan = {
#     'Selector': {
#         'task': task2,
#         'header': ['fedavg']
#     },
#     'Painter': {
#         'Curve': [
#             {'args': {'x': 'communication_round', 'y': 'val_loss'}, 'fig_option': {'title': 'valid loss on CRF10'}},
#             {'args': {'x': 'communication_round', 'y': 'val_accuracy'},
#              'fig_option': {'title': 'valid accuracy on MNIST'}},
#         ]
#     }
# }
# al.show(analysis_plan)


import json

# 假设 JSON 文件名为 data.json，包含如下内容：
# {
#     "name": "John",
#     "age": 30,
#     "city": "New York",
#     "grades": [85, 92, 89, 78]
# }

import os
import json
import matplotlib.pyplot as plt

folder_path = 'stralessRecord'  # 文件夹路径

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
            if 'test_accuracy' in data:
                fixed_field_value = data['test_accuracy']
                record_list.append(fixed_field_value)




# 假设data是你的包含8个列表的列表
import matplotlib.pyplot as plt

# 假设data是你的包含8个列表的列表
line_names = [
    '0_staleness', '0.3_staleness', '0.4_staleness', '0.5_staleness',
    '0.6_staleness', '0.7_staleness', '0.8_staleness', '0.9_staleness'
]

# 创建一个新的图形对象和一个轴对象
fig, ax = plt.subplots(figsize=(8, 6))

# 在同一个轴对象上绘制每个子列表的折线图
for i, series in enumerate(record_list):
    ax.plot(series, label=line_names[i])

# 添加图例
ax.legend()

# 添加标题和轴标签
ax.set_xlabel('Communication Round')
ax.set_ylabel('Test_Accuracy')

# 显示图形
plt.show()


