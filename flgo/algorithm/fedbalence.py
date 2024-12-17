import torch

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
        '''
        fedbalance  超参数：
        
        '''
        self.alpha=0.85
        self.agg_num=20

    def sample_balance(self):
        """
        Sample clients under the limitation of the maximum numder of concurrent clients.
        Returns:
            Selected clients.
        """
        all_clients = [cid for cid in range(self.num_clients)]
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
        self.selected_clients = self.sample_balance()
        #如果有客户端被选中，记录一条日志，显示被选中的客户端及当前时间
        self.model._round = self.current_round  #将模型的 _round 属性设置为当前的回合数
        client_id = self.selected_clients[0]
        fs_tag = self.compute_fs(client_id)

        received_packages = self.communicate(self.selected_clients, None, 2)
        client_model = received_packages['model']
        client_lambda = self.compute_lambda(client_model)
        lh_tag = self.compute_hl(client_id, client_lambda)
        self.current_round = self.current_round + 1
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

        selected = None
        # 使用 numpy 的 choice 函数进行不重复抽样
        while selected is None or selected[0] in self.concurrent_clients:
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
        self.communicate(all_clients, self.model, 1)  #下发所有模型到客户端上
        self.gv.logger.info("--------------全局模型下发--------------")
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
        l2_norm_squared = self.compute_l2_difference(w_global, w_local)

        return abs(l2_norm_squared)

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
        agg_num=self.agg_num
        if lh_tag == "H" and fs_tag == "S":
            if len(self.fl_queue) != 0:
                th = self.fl_queue.pop()
                client2_id = th["client_id"]
                client2_model = th["model"]
                self.communicate([client_id], client2_model, 1)
                self.communicate([client2_id], model, 1)
                self.concurrent_clients.difference_update([client_id,client2_id])
                return False
            else:
                self.sh_queue.append({"client_id": client_id, "model": model})
                return False
        elif lh_tag == "H" and fs_tag == "F":
            if len(self.fh_queue) != 0:
                th = self.fh_queue.pop()
                client2_id = th["client_id"]
                client2_model = th["model"]
                self.communicate([client_id], client2_model, 1)
                self.communicate([client2_id], model, 1)
                self.concurrent_clients.difference_update([client_id, client2_id])
                return False
            elif len(self.fl_queue) != 0:
                th = self.fl_queue.pop()
                client2_id = th["client_id"]
                client2_model = th["model"]
                self.communicate([client_id], client2_model, 1)
                self.communicate([client2_id], model, 1)
                self.concurrent_clients.difference_update([client_id, client2_id])
                return False
            else:
                self.fh_queue.append({"client_id": client_id, "model": model})
                return False
        elif lh_tag == "L" and fs_tag == "S":
            #1、将模型入队，如果模型数量长度
            if len(self.sl_queue) + len(self.fl_queue) == agg_num:
                self.sl_queue.append({"client_id": client_id, "model": model})
                id_list, model_list = self.queue_pop(self.sl_queue, self.fl_queue)
                agg_model=self.fedbalance_aggregate(model_list,id_list)
                self.model=agg_model
                self.communicate(id_list,agg_model,1)
                self.concurrent_clients.difference_update(id_list)
                return True
            else:
                self.sl_queue.append({"client_id": client_id, "model": model})
                return False
        else:
            if len(self.sh_queue) != 0:
                th = self.sh_queue.pop()
                client2_id = th["client_id"]
                client2_model = th["model"]
                self.communicate([client_id], client2_model, 1)
                self.communicate([client2_id], model, 1)
                self.concurrent_clients.difference_update([client_id, client2_id])
                return False
            elif len(self.fh_queue) != 0:
                th = self.fh_queue.pop()
                client2_id = th["client_id"]
                client2_model = th["model"]
                self.communicate([client_id], client2_model, 1)
                self.communicate([client2_id], model, 1)
                self.concurrent_clients.difference_update([client_id, client2_id])
                return False
                # 1、将模型入队，如果模型数量长度
            elif len(self.sl_queue) + len(self.fl_queue) == agg_num:
                self.fl_queue.append({"client_id": client_id, "model": model})
                id_list, model_list = self.queue_pop(self.sl_queue, self.fl_queue)
                agg_model = self.fedbalance_aggregate(model_list, id_list)
                self.model=agg_model
                self.communicate(id_list, agg_model, 1)
                self.concurrent_clients.difference_update(id_list)
                return True
            else:
                self.fl_queue.append({"client_id": client_id, "model": model})
                return False

    def fedbalance_aggregate(self, models, client_ids):
        """
        聚合多个客户端的模型，并将结果与 self.model 进行加权聚合。

        参数：
            models (list): 包含多个客户端模型参数的列表（PyTorch tensors）。
            client_ids (list): 包含多个客户端标识符的列表。

        返回：
            dict: 聚合后的最终模型。
        """
        # 计算所有客户端的数据量总和
        total_data_vol = sum([self.clients[client_id].datavol for client_id in client_ids])

        # 计算每个客户端模型的权重
        weights = [self.clients[client_id].datavol / total_data_vol for client_id in client_ids]

        # 加权聚合所有模型
        weighted_models = [weight * model for weight, model in zip(weights, models)]
        aggregated_model = fmodule._model_sum(weighted_models)

        # 将 aggregated_model 与 self.model 进行加权聚合
        weighted_aggregated_model = aggregated_model * (1-self.alpha)
        weighted_self_model = self.model * self.alpha
        final_model = fmodule._model_sum([weighted_aggregated_model, weighted_self_model])

        return final_model

    def communicate(self, client_id_list, send_model, mtype):
        for client_id in client_id_list:
            server_pkg = self.fedbalance_pack_model(send_model)  #全局模型
            server_pkg['__mtype__'] = mtype
            client_model = self.communicate_with(client_id, package=server_pkg)  # 与客户端进行通信，下发全局模型到本机训练，获取客户端上的模型
            if mtype == 2:
                return client_model

    def fedbalance_pack_model(self, send_model):
        return {
            "model": copy.deepcopy(send_model),
        }

    def compute_l2_difference(self, model1, model2):
        # 获取模型的所有参数
        params1 = [p.data for p in model1.parameters()]
        params2 = [p.data for p in model2.parameters()]

        # 用来存储总的 L2 范数差异
        total_l2_diff = 0.0

        # 对每一层的权重和偏置计算 L2 范数差异
        for p1, p2 in zip(params1, params2):
            # 计算 L2 范数差异并加到总差异中
            l2_diff = torch.norm(p1 - p2, p=2)  # L2范数
            total_l2_diff += l2_diff.item() ** 2  # L2范数的平方和

        # 取平方根得到最终的 L2 范数差异
        total_l2_diff = torch.sqrt(torch.tensor(total_l2_diff))
        return total_l2_diff

    def queue_pop(self, sl_queue, fl_queue):
        id_list = []
        model_list = []
        # 处理 self.sl_queue
        for item in self.sl_queue:
            id_list.append(item['client_id'])
            model_list.append(item['model'])

        # 处理 self.fl_queue
        for item in self.fl_queue:
            id_list.append(item['client_id'])
            model_list.append(item['model'])

        # 清空队列
        self.sl_queue.clear()
        self.fl_queue.clear()
        return id_list, model_list


class Client(BasicClient):
    def __init__(self, option={}):
        super().__init__(option)
        self.mu = None

    def initialize(self, *args, **kwargs):
        self.actions = {0: self.reply, 1: self.send_model, 2: self.model_train}
        self.mu = 0.1

    def send_model(self, servermodel):
        model = self.unpack(servermodel)
        self.model = model

    def model_train(self, servermodel):
        model = self.model
        model.train()
        optimizer = self.calculator.get_optimizer(model, lr=self.learning_rate, weight_decay=self.weight_decay,
                                                  momentum=self.momentum)
        for iter in range(self.num_epochs):
            # get a batch of data
            batch_data = self.get_batch_data()
            model.zero_grad()
            # calculate the loss of the model on batched dataset through task-specified calculator
            loss = self.calculator.compute_loss(model, batch_data)['loss']
            loss.backward()
            if self.clip_grad > 0: torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                                                  max_norm=self.clip_grad)
            optimizer.step()
        cpkg = self.pack(self.model)
        return cpkg
