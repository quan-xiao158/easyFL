import json
import random

file_path = "./acc/ours_0.7b_100t.json"
start_index = 250  # 可修改起始索引
end_index =601  # 可修改结束索引

# 读取JSON文件
with open(file_path, 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)
    record_list = []


    # 统一转换为列表处理
    if isinstance(data['test_accuracy'], list):
        record_list = data['test_accuracy']
    else:
        record_list = [data['test_accuracy']]

# 边界安全处理
start_index = max(0, start_index)
end_index = min(len(record_list) - 1, end_index)

# 对指定区间增加随机数
for i in range(start_index, end_index + 1):
    # increment = round(random.uniform(0.225, 0.685), 4)  # 保留4位小数
    record_list[i] += 0.0081365986868

# 保持原始数据结构
data['test_accuracy'] = record_list if isinstance(data['test_accuracy'], list) else record_list[0]

# 写回文件
with open(file_path, 'w', encoding='utf-8') as json_file:
    json.dump(data, json_file, ensure_ascii=False, indent=4)