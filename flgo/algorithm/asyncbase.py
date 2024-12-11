from flgo.algorithm.fedbase import BasicServer
from flgo.algorithm.fedbase import BasicClient as Client
import numpy as np


class AsyncServer(BasicServer):
    def __init__(self, option={}):
        super(AsyncServer, self).__init__(option)
        self.concurrent_clients = set()#正在被选择的客户端
        self.buffered_clients = set()#缓冲的客户端

    def sample(self):
        """
        Sample clients under the limitation of the maximum numder of concurrent clients.
        Returns:
            Selected clients.
        """
        all_clients = self.available_clients if 'available' in self.sample_option else [cid for cid in
                                                                                        range(self.num_clients)]
        all_clients = list(set(all_clients).difference(self.buffered_clients))
        '''
        排除已缓冲的客户端,确保在本次采样过程中不选择那些已经在 self.buffered_clients 中的客户端。
self.buffered_clients 包含那些已经被选中并正在等待处理的客户端，以防止这些客户端被再次选中，避免资源冲突或重复处理。
        '''

        clients_per_round = self.clients_per_round - len(self.concurrent_clients)#计算本轮需要选择的客户端数量
        if clients_per_round <= 0: return []#判断是否需要选择新的客户端
        clients_per_round = max(min(clients_per_round, len(all_clients)), 1)
        # full sampling with unlimited communication resources of the server
        if 'full' in self.sample_option:
            return all_clients
        # sample clients
        elif 'uniform' in self.sample_option:
            # original sample proposed by fedavg
            selected_clients = list(np.random.choice(all_clients, clients_per_round, replace=False)) if len(
                all_clients) > 0 else []
        elif 'md' in self.sample_option:
            # the default setting that is introduced by FedProx, where the clients are sampled with the probability in proportion to their local_movielens_recommendation data sizes
            local_data_vols = [self.clients[cid].datavol for cid in all_clients]
            total_data_vol = sum(local_data_vols)
            p = np.array(local_data_vols) / total_data_vol
            selected_clients = list(np.random.choice(all_clients, clients_per_round, replace=True, p=p)) if len(
                all_clients) > 0 else []
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
