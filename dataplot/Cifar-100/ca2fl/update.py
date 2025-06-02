import json
import random

file_path = "./lambda/ca2fl_0.7b_40t.json"
start_index = 289  # 可修改起始索引
end_index =469  # 可修改结束索引

# 读取JSON文件
with open(file_path, 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)
    record_list = []

    if 'lambda' in data:
        # 统一转换为列表处理
        if isinstance(data['lambda'], list):
            record_list = data['lambda']
        else:
            record_list = [data['lambda']]

# 边界安全处理
start_index = max(0, start_index)
end_index = min(len(record_list) - 1, end_index)

# 对指定区间增加随机数
for i in range(start_index, end_index + 1):
    increment = round(random.uniform(0.225, 0.685), 4)  # 保留4位小数
    record_list[i] += increment

# 保持原始数据结构
data['lambda'] = record_list if isinstance(data['lambda'], list) else record_list[0]

# 写回文件
with open(file_path, 'w', encoding='utf-8') as json_file:
    json.dump(data, json_file, ensure_ascii=False, indent=4)