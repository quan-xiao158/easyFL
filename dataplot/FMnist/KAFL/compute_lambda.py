import json
import math
import os

import numpy as np


# 示例数据

with open('./KAFL/lambda/lr_KAFL_0.7b_10t.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
data=data['lambda']
# 计算均值
mean_value = np.mean(data)

# 输出结果
print(f"均值: {mean_value}")