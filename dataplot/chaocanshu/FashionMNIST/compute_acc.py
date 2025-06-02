import json
import math
import os

import numpy as np


# 示例数据

with open('ρ=0.25_β=0.8.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
data=data['test_accuracy'][950:1000]
mean_value = np.mean(data)
sample_variance = math.sqrt(np.var(data, ddof=1))  # ddof=1 表示使用 n-1 作为分母
print(f"均值: {mean_value}")
print(f"样本方差: {sample_variance}")