import json
import math

import torch
from torch import cosine_similarity

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
        """
        Running the FL symtem where the global model is trained and evaluated iteratively.
        """

        self.gv.logger.time_start('Total Time Cost')
        all_clients = self.available_clients if 'available' in self.sample_option else [cid for cid in
                                                                                        range(self.num_clients)]
        self.weights=self.generate_weight(all_clients)
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
            if updated is True and self.option['plot'] is True:
                self.Record_with_test()
            elif updated is True and self.option['plot'] is False:
                self.Record_without_test()
            else:continue
        self.gv.logger.info("=================End==================")
        self.gv.logger.time_end('Total Time Cost')
        # save results as .json file
        self.gv.logger.save_output_as_json()
        return
    def Record_with_test(self):
        self.gv.logger.info("------Round {}:{}{}b{}t--------------".format(self.round_number,self.option["algorithm"],self.option["b"],self.option["t"]))
        # check log interval
        if self.gv.logger.check_if_log(self.current_round, self.eval_interval):
            self.gv.logger.time_start('Eval Time Cost')
            self.gv.logger.log_once()  # 验证模型损失和准确率
            self.gv.logger.time_end('Eval Time Cost')
            self._save_checkpoint()
        # check if early stopping
        self.current_round += 1
        # decay learning rate
        self.global_lr_scheduler(self.current_round)
        if self.round_number % 50 == 0:
            self.gv.logger.save_output_as_json()
    def Record_without_test(self):
        self.gv.logger.info(
            "------Round {}:{}{}b{}t--------------".format(self.round_number, self.option["algorithm"], self.option["b"],
                                                         self.option["t"]))
        # check log interval
        if self.gv.logger.check_if_log(self.current_round, self.eval_interval):
            if self.num_rounds-self.current_round<=30:
                self.gv.logger.time_start('Eval Time Cost')
                self.gv.logger.log_once()  # 验证模型损失和准确率
                self.gv.logger.time_end('Eval Time Cost')
                self._save_checkpoint()
        # check if early stopping
        self.current_round += 1
        # decay learning rate
        self.global_lr_scheduler(self.current_round)
        if self.round_number % 50 == 0:
            self.gv.logger.save_output_as_json()
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

    import numpy as np

    import numpy as np

    def sample_one_client(self, all_clients):


        selected = np.random.choice(all_clients, size=1, replace=False, p=self.weights)
        return selected[0]

    def computeDifference(self):
        all_clients = [cid for cid in range(self.num_clients)]
        gmodel = self.model
        received_packages = self.communicate(all_clients, self.model, 3)
        models = []
        for index in received_packages:
            models.append(index['model'])
        # 从模型实例中提取参数并展平
        global_vec = torch.cat([t.flatten() for t in gmodel.state_dict().values()])

        similarities = []
        for model, id in zip(models, all_clients):
            local_vec = torch.cat([t.flatten() for t in model.state_dict().values()])
            cos_sim = cosine_similarity(global_vec.unsqueeze(0), local_vec.unsqueeze(0), dim=1)
            similarities.append(
                (1 - cos_sim.item()) * (1 + 0.5 * math.log(1 + (self.current_round - self.server_send_round[id]))))

        avg_sim = sum(similarities) / len(similarities)

        # 追加到JSON文件
        filename = "{}{}b{}t.json".format(self.option['algorithm'],self.option['b'],self.option['t'])
        try:
            with open(filename, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            data = {"lambda":[]}

        data["lambda"].append(avg_sim)
        with open(filename, "w") as f:
            json.dump(data, f)

    def generate_weight(self,all_clients):
        b = 1 - self.option['b']
        t = self.option['t']
        if len(all_clients) != 100:
            raise ValueError("all_clients 的长度必须为100。")
        if not 0 <= b <= 1:
            raise ValueError("参数 b 必须在 [0, 1] 范围内。")
        if t < 0:
            raise ValueError("参数 t 不能为负数。")

        # 生成随机分组索引

        k = int(b * len(all_clients))

        # 计算分母确保数值稳定性
        denominator = k * t + (len(all_clients) - k)
        if denominator <= 0:
            raise ValueError("无效的权重分配组合，请检查参数 t 和 b 的值。")
        if self.option['client_weight']=='random':
            client_indices = np.random.permutation(len(all_clients))
            # 初始化权重并分配
            weights = np.zeros(len(all_clients))
            if k > 0:
                weights[client_indices[:k]] = t / denominator  # 随机选择的前b比例部分
            if (len(all_clients) - k) > 0:
                weights[client_indices[k:]] = 1 / denominator  # 剩余的随机部分
        else:
            weights = np.zeros(len(all_clients))
            if k > 0:
                weights[:k] = t / denominator
            if (len(all_clients) - k) > 0:
                weights[k:] = 1 / denominator
        # 强制归一化处理（应对浮点运算误差）
        weights /= weights.sum()
        return  weights