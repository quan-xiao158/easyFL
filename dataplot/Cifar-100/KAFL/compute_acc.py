import json
import math
import os

import numpy as np


# 示例数据

with open('./acc/KAFL_0.7b_10t.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
data=data['test_accuracy'][570:600]
# 计算均值
mean_value = np.mean(data)

# 计算总体方差

# 计算样本方差
sample_variance = math.sqrt(np.var(data, ddof=1))  # ddof=1 表示使用 n-1 作为分母

# 输出结果
print(f"均值: {mean_value}")
print(f"样本方差: {sample_variance}")