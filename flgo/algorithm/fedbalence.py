from flgo.utils import fmodule
from flgo.algorithm.asyncbase import AsyncServer
from flgo.algorithm.fedbase import BasicServer, BasicClient
from flgo.algorithm.fedprox import Client
import numpy as np
from collections import deque
import copy
import flgo.simulator.base as ss


class Server(AsyncServer):
    def __init__(self, option={}):
        super(Server, self).__init__(option)
        self.concurrent_clients = set()  #正在被选择的客户端
        self.buffered_clients = set()  #缓冲的客户端
        self.server_send_round = [0] * 100  # 服务器下发模型round记录
        self.client_fs_list = [-1] * 100  #服务器记录客户端时间差列表
        self.client_lh_list = [-1] * 100  #服务器记录客户端上传模型陈旧度列表
        self.fl_queue = deque()
        self.fh_queue = deque()
        self.sl_queue = deque()
        self.sh_queue = deque()

    def sample(self):
        """
        Sample clients under the limitation of the maximum numder of concurrent clients.
        Returns:
            Selected clients.
        """
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

    def iterate(self):
        """
        1、sample()采样出两个客户端，模拟异步下有两个客户端提交模型，得到客户端编号列表，selected_clients
        2、communicate（）与客户端通信，得到客户端上的两个模型，received_packages
        4、package_handler（）对模型进行聚合（并下发）
        5、返回全局模型是否更新
        """
        self.selected_clients = self.sample()
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
        '''
        更新时延列表
        '''
        client_id = self.selected_clients[0]
        fs_tag = self.compute_fs(client_id)

        received_packages = self.communicate(self.selected_clients, asynchronous=True)
        #选定的客户端进行通信，发送当前的全局模型并接收客户端的更新。asynchronous=True 表明这是一次异步通信，不会阻塞等待所有客户端的响应。
        #received_packages 是一个包含来自客户端的更新数据的字典，特别是客户端的标识符 __cid。
        client_model = received_packages['model'][0]
        client_lambda = self.compute_lambda(client_model)
        lh_tag = self.compute_hl(client_id, client_lambda)
        return self.core_select_algorithm(client_id, fs_tag, lh_tag, client_model)

    def select_two_clients_by_group(self, all_clients):
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
        selected = np.random.choice(all_clients, size=1, replace=False, p=weights)

        return selected.tolist()

    def run(self):
        """
        Running the FL symtem where the global model is trained and evaluated iteratively.
        """
        self.gv.logger.time_start('Total Time Cost')
        if not self._load_checkpoint() and self.eval_interval > 0:
            # evaluating initial model performance
            self.gv.logger.info("--------------Initial Evaluation--------------")
            self.gv.logger.time_start('Eval Time Cost')
            self.gv.logger.log_once()  # 初始评估全局模型的准确度
            self.gv.logger.time_end('Eval Time Cost')
        all_clients = self.available_clients if 'available' in self.sample_option else [cid for cid in
                                                                                        range(self.num_clients)]
        all_clients = list(set(all_clients).difference(self.buffered_clients))
        self.communicate(all_clients,self.model,1)
        while True:
            if self._if_exit(): break
            self.gv.clock.step()
            # iterate
            updated = self.iterate()  # 进行客户端选择迭代训练和模型聚合
            # using logger to evaluate the model if the model is updated
            if updated is True:
                self.gv.logger.info("--------------Round {}--------------".format(self.current_round))
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
            else:
                self.gv.logger.info("进行模型交换")
        self.gv.logger.info("=================End==================")
        self.gv.logger.time_end('Total Time Cost')
        # save results as .json file
        self.gv.logger.save_output_as_json()
        return

    def median(self, lst):
        new_lst = [x for x in lst if x != -1]
        new_lst.sort()
        n = len(new_lst)
        if n % 2 == 0:
            mid1 = new_lst[n // 2 - 1]
            mid2 = new_lst[n // 2]
            median = (mid1 + mid2) / 2
        else:
            median = new_lst[n // 2]
        return median

    def compute_lambda(self, w_local):
        w_global = self.model
        dot_product = np.dot(w_local, w_global)

        # 计算全局模型的 L2 范数平方
        l2_norm_squared = np.linalg.norm(w_global) ** 2

        # 计算 λ_i
        lambda_i = (dot_product / l2_norm_squared) - 1

        return abs(lambda_i)

    def compute_hl(self, client_id, lambda_i):
        self.client_lh_list[client_id] = lambda_i
        filtered_list = list(filter(lambda x: x != -1, self.client_lh_list))

        # 计算均值
        mean_value = sum(filtered_list) / len(filtered_list) if filtered_list else 0
        if lambda_i <= mean_value:
            return "L"
        else:
            return "H"

    def compute_fs(self, client_id):
        round_send = self.server_send_round[client_id]
        current_round = self.current_round
        time_delay = current_round - round_send
        self.server_send_round[self.selected_clients[0]] = self.current_round
        self.client_fs_list[client_id] = time_delay
        media = self.median(self.client_fs_list)

        if time_delay <= media:
            return "F"
        else:
            return "S"

    def core_select_algorithm(self, client_id, fs_tag, lh_tag, model):
        if lh_tag == "S" and fs_tag == "H":
            if len(self.fl_queue) != 0:
                th = self.fl_queue.pop()
                client2_id = th["client_id"]
                client2_model = th["model"]
                self.communicate(client_id, client2_model)
                self.communicate(client2_id, model)
                return False
            else:
                self.sh_queue.append({"client_id": client_id, "model": model})
                return False
        elif lh_tag == "F" and fs_tag == "H":
            if len(self.fh_queue) != 0:
                th = self.fh_queue.pop()
                client2_id = th["client_id"]
                client2_model = th["model"]
                self.communicate(client_id, client2_model)
                self.communicate(client2_id, model)
                return False
            elif len(self.fl_queue) != 0:
                th = self.fl_queue.pop()
                client2_id = th["client_id"]
                client2_model = th["model"]
                self.communicate(client_id, client2_model)
                self.communicate(client2_id, model)
                return False
            else:
                self.fh_queue.append({"client_id": client_id, "model": model})
                return False
        elif lh_tag == "S" and fs_tag == "L":
            if len(self.fl_queue) != 0:
                th = self.fl_queue.pop()
                client2_id = th["client_id"]
                client2_model = th["model"]
                self.model = self.fedbalance_aggregate(model, client2_model, client_id, client2_id)
                return True
            elif len(self.sl_queue) != 0:
                th = self.sl_queue.pop()
                client2_id = th["client_id"]
                client2_model = th["model"]
                self.model = self.fedbalance_aggregate(model, client2_model, client_id, client2_id)
                return True
            else:
                self.sl_queue.append({"client_id": client_id, "model": model})
                return False
        else:
            if len(self.sh_queue) != 0:
                th = self.sh_queue.pop()
                client2_id = th["client_id"]
                client2_model = th["model"]
                self.communicate(client_id, client2_model)
                self.communicate(client2_id, model)
                return False
            elif len(self.fh_queue) != 0:
                th = self.fl_queue.pop()
                client2_id = th["client_id"]
                client2_model = th["model"]
                self.communicate(client_id, client2_model)
                self.communicate(client2_id, model)
                return False
            elif len(self.sl_queue) != 0:
                th = self.sl_queue.pop()
                client2_id = th["client_id"]
                client2_model = th["model"]
                self.model = self.fedbalance_aggregate(model, client2_model, client_id, client2_id)
                return True
            elif len(self.fl_queue) != 0:
                th = self.fl_queue.pop()
                client2_id = th["client_id"]
                client2_model = th["model"]
                self.model = self.fedbalance_aggregate(model, client2_model, client_id, client2_id)
                return True
            else:
                self.fl_queue.append({"client_id": client_id, "model": model})
                return False

    def fedbalance_aggregate(self, model1, model2, client1_id, client2_id):
        """
        聚合两个客户端的模型，并将结果与 self.model 进行 1:1 的聚合。

        参数：
            model1 (dict): 第一个客户端的模型参数（PyTorch tensors）。
            model2 (dict): 第二个客户端的模型参数（PyTorch tensors）。
            client1_id (int/str): 第一个客户端的标识符。
            client2_id (int/str): 第二个客户端的标识符。

        返回：
            dict: 聚合后的最终模型。
        """
        # 获取两个客户端的数据量
        data_vol1 = self.clients[client1_id].datavol
        data_vol2 = self.clients[client2_id].datavol
        total_data_vol = data_vol1 + data_vol2

        # 计算权重
        p1 = data_vol1 / total_data_vol
        p2 = data_vol2 / total_data_vol

        # 加权聚合 model1 和 model2
        weighted_model1 = {key: value * p1 for key, value in model1.items()}
        weighted_model2 = {key: value * p2 for key, value in model2.items()}
        aggregated_model = fmodule._model_sum([weighted_model1, weighted_model2])

        # 将 aggregated_model 与 self.model 进行 1:1 聚合
        weighted_aggregated_model = {key: value * 0.5 for key, value in aggregated_model.items()}
        weighted_self_model = {key: value * 0.5 for key, value in self.model.items()}
        final_model = fmodule._model_sum([weighted_aggregated_model, weighted_self_model])

        return final_model

    @ss.with_clock
    def communicate(self, client_id_list, model, mtype, asynchronous=False):
        for client_id in client_id_list:
            server_pkg = self.fedbalance_pack_model(client_id, model, mtype)  #全局模型
            server_pkg['__mtype__'] = mtype
            self.communicate_with(client_id, package=server_pkg)  # 与客户端进行通信，下发全局模型到本机训练，获取客户端上的模型

    def fedbalance_pack_model(self, client_id, send_model, mtype=0, *args, **kwargs):
        return {
            "model": copy.deepcopy(send_model),
        }


class Client(BasicClient):
    def initialize(self, *args, **kwargs):
        self.actions = {0: self.reply, 1: self.send_model, 2: self.model_train}

    def send_model(self, servermodel):
        self.model = servermodel

    def model_train(self):
        self.train(self.model)
        cpkg = self.pack(self.model)
        return cpkg
