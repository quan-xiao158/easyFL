import numpy as np

# 假设第i个边缘设备的本地模型在版本j时的权重矩阵 w_i^l(j)
w_i_l_j = np.array([[1, 2], [3, 4]])
# 第t轮时的全局模型权重矩阵 w^g(t)
w_g_t = np.array([[5, 6], [7, 8]])

# 计算分子：矩阵内积（这里其实是Frobenius内积，对应向量内积的矩阵推广）
inner_product = np.sum(w_i_l_j * w_g_t)

# 计算分母：全局模型矩阵的Frobenius范数的平方
norm_squared = np.linalg.norm(w_g_t, 'fro') ** 2

# 计算lambda_i
lambda_i = (inner_product / norm_squared) - 1

print(lambda_i)