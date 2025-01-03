from flgo.algorithm.fedbase import BasicServer
from flgo.algorithm.fedbase import BasicClient as Client
import numpy as np


class AsyncServer(BasicServer):
    def __init__(self, option={}):
        super(AsyncServer, self).__init__(option)
        self.concurrent_clients = set()  #正在被选择的客户端
        self.buffered_clients = set()  #缓冲的客户端
        self.selected_client = None
        self.selected_clients=[]
        self.buff_len=15
        self.total_traffic = 0
        self.additional_traffic = 0

    def sample(self):
        all_clients = self.available_clients if 'available' in self.sample_option else [cid for cid in
                                                                                        range(self.num_clients)]
        all_clients = list(set(all_clients).difference(self.buffered_clients))
        selected_clients = self.select_two_clients_by_group(all_clients)
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

    def run(self):
        np.random.seed(42)
        """
        Running the FL symtem where the global model is trained and evaluated iteratively.
        """
        self.gv.logger.time_start('Total Time Cost')
        all_clients = self.available_clients if 'available' in self.sample_option else [cid for cid in
                                                                                        range(self.num_clients)]
        all_clients = list(set(all_clients).difference(self.buffered_clients))
        self.gv.logger.info("--------------Initial Evaluation--------------")
        self.gv.logger.time_start('Eval Time Cost')
        self.gv.logger.log_once()  # 初始评估全局模型的准确度
        self.gv.logger.time_end('Eval Time Cost')
        self.model._round = 0  # 将模型的 _round 属性设置为当前的回合数
        self.communicate(all_clients, self.model, 1)  # 下发所有模型到客户端上
        self.gv.logger.info("--------------全局模型下发--------------")
        while True:
            if self.round_number > self.num_rounds: break
            self.gv.clock.step()
            # iterate
            updated = self.iterate()  # 进行客户端选择迭代训练和模型聚合
            # using logger to evaluate the model if the model is updated
            if updated is True:
                self.gv.logger.info("--------------Round {}--------------".format(self.round_number))
                # check log interval
                if self.gv.logger.check_if_log(self.current_round, self.eval_interval):
                    self.gv.logger.time_start('Eval Time Cost')
                    self.gv.logger.log_once()  # 验证模型损失和准确率
                    self.gv.logger.time_end('Eval Time Cost')
                    self._save_checkpoint()
                # check if early stopping
                if self.gv.logger.early_stop(): break
                self.current_round += 1
                # decay learning rate
                self.global_lr_scheduler(self.current_round)
                if self.round_number % 50 == 0:
                    self.gv.logger.save_output_as_json()
                # self.gv.logger.info("总通信量{}额外通信量{}".format(self.total_traffic, self.additional_traffic))
        self.gv.logger.info("=================End==================")
        self.gv.logger.time_end('Total Time Cost')
        # save results as .json file
        self.gv.logger.save_output_as_json()
        return
    def iterate(self):
        """
        作用：定义了服务器在每个时刻执行的迭代过程。
        The procedure of the server at each moment. Compared to synchronous methods, asynchronous servers perform iterations in a time-level view instead of a round-level view.

        Returns:
            一个布尔值，指示当前迭代是否更新了全局模型。
            is_model_updated (bool): True if the global model is updated at the current iteration
        """
        self.selected_clients = self.sample()
        self.concurrent_clients.update(
            set(self.selected_clients))
        if len(self.selected_clients) > 0: self.gv.logger.info(
            'Select clients {} at time {}.'.format(self.selected_clients, self.gv.clock.current_time))

        self.model._round = self.current_round

        received_packages = self.communicate(self.selected_clients,
                                             asynchronous=True)
        self.concurrent_clients.difference_update(
            set(received_packages['__cid']))
        self.buffered_clients.update(
            set(received_packages['__cid']))
        if len(received_packages['__cid']) > 0: self.gv.logger.info(
            'Receive new models from clients {} at time {}'.format(received_packages['__cid'],
                                                                   self.gv.clock.current_time))
        is_model_updated = self.package_handler(received_packages)
        if is_model_updated: self.buffered_clients = set()
        return is_model_updated

    def sample_async(self):
        """
        Sample clients under the limitation of the maximum numder of concurrent clients.
        Returns:
            Selected clients.
        """
        all_clients = [cid for cid in range(self.num_clients)]
        all_clients = list(set(all_clients).difference(self.buffered_clients))
        selected_clients = self.sample_one_client(all_clients)
        return selected_clients

    def sample_one_client(self, all_clients):
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
            {'range': range(0, 10), 'prob': 1},
            {'range': range(10, 20), 'prob': 0.9},
            {'range': range(20, 30), 'prob': 0.8},
            {'range': range(30, 40), 'prob': 0.7},
            {'range': range(40, 50), 'prob': 0.6},
            {'range': range(50, 60), 'prob': 0.5},
            {'range': range(60, 70), 'prob': 0.4},
            {'range': range(70, 80), 'prob': 0.3},
            {'range': range(80, 90), 'prob': 0.2},
            {'range': range(90, 100), 'prob': 0.1},
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

        selected = None
        # 使用 numpy 的 choice 函数进行不重复抽样
        while selected is None or selected[0] in self.concurrent_clients:
            selected = np.random.choice(all_clients, size=1, replace=False, p=weights)
        return selected[0]
