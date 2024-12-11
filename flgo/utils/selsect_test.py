import numpy as np


def select_two_clients_by_group(all_clients):
    """
    按照每25个为一组，并根据指定的组概率从all_clients中选择两个不同的客户端。

    组定义：
    - 组1: ID 0-24，概率 0.4
    - 组2: ID 25-49，概率 0.3
    - 组3: ID 50-74，概率 0.2
    - 组4: ID 75-99，概率 0.1

    参数：
    all_clients (list): 包含所有100个客户端的列表。

    返回：
    list: 选中的两个不同客户端的列表。
    """
    if len(all_clients) != 100:
        raise ValueError("all_clients 的长度必须为100。")

    # 定义组及其对应的概率
    groups = [
        {'range': range(0, 25), 'prob': 0.4},  # 组1: ID 0-24
        {'range': range(25, 50), 'prob': 0.3},  # 组2: ID 25-49
        {'range': range(50, 75), 'prob': 0.2},  # 组3: ID 50-74
        {'range': range(75, 100), 'prob': 0.1}  # 组4: ID 75-99
    ]

    # 初始化权重列表
    weights = np.zeros(len(all_clients))

    # 分配权重
    for group in groups:
        group_range = group['range']
        group_prob = group['prob']
        group_size = len(group_range)
        if group_prob > 0:
            # 每个客户端的权重 = 组概率 / 组大小
            weights[list(group_range)] = group_prob / group_size
        # 如果组概率为0，则对应客户端权重保持为0

    # 确保权重之和为1
    total_weight = weights.sum()
    if not np.isclose(total_weight, 1.0):
        # 归一化权重
        weights = weights / total_weight

    # 使用 numpy 的 choice 函数进行不重复抽样
    selected = np.random.choice(all_clients, size=2, replace=False, p=weights)

    return selected.tolist()


# 示例用法
if __name__ == "__main__":
    for i in range(20):
        all_clients = list(range(100))  # 客户端ID 0-99

        selected_clients = select_two_clients_by_group(all_clients)
        print("选中的两个客户端:", selected_clients)
