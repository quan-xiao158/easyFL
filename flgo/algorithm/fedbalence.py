from flgo.algorithm.fedbase import BasicServer
from flgo.algorithm.fedbase import BasicClient as Client
import numpy as np


class FedbalenceServer(BasicServer):
    def __init__(self, option={}):
        super(FedbalenceServer, self).__init__(option)
        self.concurrent_clients = set()#正在被选择的客户端
        self.buffered_clients = set()#缓冲的客户端

    def sample(self):
        """
        Sample clients under the limitation of the maximum numder of concurrent clients.
        Returns:
            Selected clients.
        """
        all_clients = self.available_clients if 'available' in self.sample_option else [cid for cid in                                                                           range(self.num_clients)]
        all_clients = list(set(all_clients).difference(self.buffered_clients))
        selected_clients=self.select_two_clients_by_group(all_clients)
        return selected_clients

    def package_handler(self, received_packages: dict):
        """
        Handle packages received from clients and return whether the global model is updated in this function.

        Args:
            received_packages (dict): a dict consisting of uploaded contents from clients
        Returns:
            is_model_updated (bool): True if the global model is updated in this function.
        """
        if self.is_package_empty(received_packages): return False
        self.model = self.aggregate(received_packages['model'])
        return True

    def is_package_empty(self, received_packages: dict):
        """
        Check whether the package dict is empty

        Returns:
            is_empty (bool): True if the package dict is empty
        """
        return len(received_packages['__cid']) == 0

    def iterate(self):
        """
        作用：定义了服务器在每个时刻执行的迭代过程。
        The procedure of the server at each moment. Compared to synchronous methods, asynchronous servers perform iterations in a time-level view instead of a round-level view.

        Returns:
            一个布尔值，指示当前迭代是否更新了全局模型。
            is_model_updated (bool): True if the global model is updated at the current iteration
        """
        self.selected_clients = self.sample()  #从可用的客户端中选择一组客户端进行当前迭代的训练。
        self.concurrent_clients.update(
            set(self.selected_clients))  #将选中的客户端添加到 self.concurrent_clients 集合中，表示这些客户端正在进行训练或通信。
        if len(self.selected_clients) > 0: self.gv.logger.info(
            'Select clients {} at time {}.'.format(self.selected_clients, self.gv.clock.current_time))
        #如果有客户端被选中，记录一条日志，显示被选中的客户端及当前时间
        self.model._round = self.current_round  #将模型的 _round 属性设置为当前的回合数
        '''
        改动，
        原来：received_packages为选择的客户端进行训练后发给服务器进行聚合
        新： received_packages为选择的客户端进行训练，服务器依据客户端性能概率挑选两个客户端进行聚合
        '''
        received_packages = self.communicate(self.selected_clients,
                                             asynchronous=True)  #选定的客户端进行通信，发送当前的全局模型并接收客户端的更新。asynchronous=True 表明这是一次异步通信，不会阻塞等待所有客户端的响应。
        #received_packages 是一个包含来自客户端的更新数据的字典，特别是客户端的标识符 __cid。
        self.concurrent_clients.difference_update(
            set(received_packages['__cid']))  #从 self.concurrent_clients 中移除已经响应的客户端。
        self.buffered_clients.update(
            set(received_packages['__cid']))  #将这些响应的客户端添加到 self.buffered_clients 集合中，表示它们的更新已被接收并正在处理。
        if len(received_packages['__cid']) > 0: self.gv.logger.info(
            'Receive new models from clients {} at time {}'.format(received_packages['__cid'],
                                                                   self.gv.clock.current_time))
        is_model_updated = self.package_handler(received_packages)  #调用 方法处理接收到的客户端更新包。这可能涉及聚合客户端的模型更新（如平均化权重）并更新全局模型。
        if is_model_updated: self.buffered_clients = set()  #如果全局模型已更新，清空 self.buffered_clients 集合。这可能意味着所有已接收的更新已被整合，准备接受新的客户端更新。
        return is_model_updated

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