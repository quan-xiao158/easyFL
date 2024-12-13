client_lh_list =[-1,-1,-1,5,4]

# 过滤掉值为-1的元素
filtered_list = list(filter(lambda x: x != -1, client_lh_list))

# 计算均值
mean_value = sum(filtered_list) / len(filtered_list) if filtered_list else 0

print(mean_value)
