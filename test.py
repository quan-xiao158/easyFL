import json
import math
import os

import numpy as np


# 示例数据

data =[        0.9484,
        0.9434,
        0.9516,
        0.952,
        0.9484,
        0.9495,
        0.951,
        0.9479,
        0.9446,
        0.9499
    ]

# 计算均值
mean_value = np.mean(data)

# 计算总体方差

# 计算样本方差
sample_variance = math.sqrt(np.var(data, ddof=1))  # ddof=1 表示使用 n-1 作为分母

# 输出结果
print(f"均值: {mean_value}")
print(f"样本方差: {sample_variance}")