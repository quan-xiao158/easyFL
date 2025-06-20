
import torch
from torch import cosine_similarity

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

        self.concurrent_clients = set()  # 正在被选择的客户端
        self.buffered_clients = set()  # 缓冲的客户端
        self.server_send_round = [0] * 100  # 服务器下发模型round记录
        self.client_fs_list = [-1] * 100  # 服务器记录客户端时间差列表
        self.client_lh_list = [-1] * 100  # 服务器记录客户端上传模型陈旧度列表
        self.decay_list = [0] * 100
        self.round_number = 0
        self.buff_queue = deque()
        '''
        fedbalance  超参数：

        '''
        self.alpha = 0.72
        self.agg_num = 10
        self.buff_len=10

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
        self.current_round += 1

        self.selected_client = self.sample_async()
        # 如果有客户端被选中，记录一条日志，显示被选中的客户端及当前时间
        self.model._round = self.current_round  # 将模型的 _round 属性设置为当前的回合数
        self.selected_clients.append(self.selected_client)
        self.concurrent_clients.add(self.selected_client)
        received_packages = self.communicate([self.selected_client], None, 2)
        for index in received_packages:
            self.buff_queue.append(index['model'])
        if len(self.buff_queue) == self.buff_len:
            model_list = []
            for i in range(self.buff_len):
                model_list.append(self.buff_queue.pop())

            self.computeDifference()
            self.model = self.async_aggregate(model_list, self.selected_clients)
            self.communicate(self.selected_clients, self.model, 1)
            self.selected_clients.clear()
            self.concurrent_clients.clear()
            self.round_number += 1
            return True
        return False

    def fedasync_late_aggregate(self, models, client_ids):
        """
        聚合来自不同客户端的模型，考虑模型更新的延迟。

        参数:
        - models (list): 模型列表，每个元素代表一个客户端的模型。
        - client_ids (list): 客户端ID列表，对应于models中的每个模型。
        - late_list (list): 延迟列表，对应于client_ids，每个元素表示该客户端模型的延迟（current_round - tau）。

        返回:
        - bool: 如果聚合成功，返回True；如果输入为空或不合法，返回False。
        """

        dl = []
        for id in self.selected_clients:
            dl.append(self.round_number - self.server_send_round[id])
            self.server_send_round[id] = self.round_number

        alpha_ts = [self.alpha *len(models)]
        # 4. 更新每个接收到的模型，与当前模型进行加权融合

        currently_updated_models = []
        for alpha_t, model_k in zip(alpha_ts, models):
            # 假设 self.model 和 model_k 支持标量乘法和加法操作
            updated_model = (1 - alpha_t) * self.model + alpha_t * model_k
            currently_updated_models.append(updated_model)

        # 5. 聚合所有更新后的模型，更新当前模型

        return self.async_aggregate(currently_updated_models, client_ids)

    def async_aggregate(self, models, client_ids):
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

        return aggregated_model

    def communicate(self, client_id_list, send_model, mtype):
        self.total_traffic += len(client_id_list)
        model_list = []
        for client_id in client_id_list:
            server_pkg = self.fedbalance_pack_model(send_model)  # 全局模型
            server_pkg['__mtype__'] = mtype
            client_model = self.communicate_with(client_id, package=server_pkg)  # 与客户端进行通信，下发全局模型到本机训练，获取客户端上的模型
            if mtype == 2 or 3:
                model_list.append(client_model)
        if mtype == 2 or 3:
            return model_list

    def fedbalance_pack_model(self, send_model):
        return {
            "model": copy.deepcopy(send_model),
        }


class Client(BasicClient):
    def __init__(self, option={}):
        super().__init__(option)
        self.mu = None

    def initialize(self, *args, **kwargs):
        self.actions = {0: self.reply, 1: self.send_model, 2: self.rp, 3: self.receive_model}
        self.mu = 0.1

    def send_model(self, servermodel):
        model = self.unpack(servermodel)
        self.model = model

    def rp(self, servermodel):
        self.model_train(self.model)
        cpkg = self.pack(self.model)
        return cpkg

    def receive_model(self, servermodel):
        return self.pack(self.model)

    @fmodule.with_multi_gpus
    def model_train(self, model):
        src_model = copy.deepcopy(model)
        src_model.freeze_grad()
        model.train()
        optimizer = self.calculator.get_optimizer(model, lr=self.learning_rate, weight_decay=self.weight_decay,
                                                  momentum=self.momentum)
        for iter in range(10):
            # get a batch of data
            batch_data = self.get_batch_data()
            self.model.zero_grad()
            # calculate the loss of the model on batched dataset through task-specified calculator
            loss = self.calculator.compute_loss(model, batch_data)['loss']
            loss_proximal = 0
            for pm, ps in zip(self.model.parameters(), src_model.parameters()):
                loss_proximal += torch.sum(torch.pow(pm - ps, 2))
            loss = loss + 0.5 * 0.005 * loss_proximal
            loss.backward()
            if self.clip_grad > 0: torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                                                  max_norm=self.clip_grad)
            optimizer.step()
