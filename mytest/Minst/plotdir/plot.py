
import os
import json
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=10)
folder_path = 'biasRecord'  # 文件夹路径
plt.rcParams['font.sans-serif']=['SimHei']
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
            if 'val_accuracy' in data:
                fixed_field_value = data['val_accuracy']
                record_list.append(fixed_field_value)




# 假设data是你的包含8个列表的列表
import matplotlib.pyplot as plt

# # 假设data是你的包含8个列表的列表
# line_names = [
#     '无陈旧度', '0.3陈旧度', '0.4陈旧度', '0.5陈旧度',
#     '0.6陈旧度', '0.7陈旧度', '0.8陈旧度', '0.9陈旧度'
# ]
line_names = [
    '0训练偏差','10训练偏差','15训练偏差','17训练偏差','20训练偏差'
]

# line_names = [
#     '0biasAndStraless','17bias','0.4straless','biasAndStraless'
# ]

# 创建一个新的图形对象和一个轴对象
fig, ax = plt.subplots(figsize=(8, 6))

# 在同一个轴对象上绘制每个子列表的折线图
for i, series in enumerate(record_list):
    ax.plot(series, label=line_names[i])

# 添加图例
ax.legend()

# 添加标题和轴标签
ax.set_xlabel('训练轮次',fontproperties=font_set)
ax.set_ylabel('测试集准确率',fontproperties=font_set)

# 显示图形
plt.show()


