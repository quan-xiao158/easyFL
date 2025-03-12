import math

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

        self.concurrent_clients = set()  # 正在被选择的客户端
        self.buffered_clients = set()  # 缓冲的客户端
        self.server_send_round = [0] * 100  # 服务器下发模型round记录
        self.commit_num = [1] * 100
        self.client_fs_list = [-1] * 100  # 服务器记录客户端时间差列表
        self.client_lh_list = [-1] * 100  # 服务器记录客户端上传模型陈旧度列表
        self.decay_list = [0] * 100
        self.lambada = [0.0] * 100
        self.round_number = 0
        self.buff_queue = deque()
        '''
        fedbalance  超参数：

        '''
        self.alpha = 0.72
        self.agg_num = 15
        self.r = 0.5

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
        self.commit_num[self.selected_client] += 1
        self.model._round = self.current_round
        self.selected_clients.append(self.selected_client)
        self.concurrent_clients.add(self.selected_client)
        received_packages = self.communicate([self.selected_client], None, 2)
        self.lambada[self.selected_client] = self.compute_lambda(received_packages[0]['model'])
        for index in received_packages:
            self.buff_queue.append(index['model'])
        if len(self.buff_queue) == self.buff_len:
            self.computeDifference()
            model_list = []
            for i in range(self.buff_len):
                model_list.append(self.buff_queue.pop())
            self.model = self.kafl_aggregate(model_list, self.selected_clients)
            self.communicate(self.selected_clients, self.model, 1)
            self.selected_clients.clear()
            self.concurrent_clients.clear()
            self.round_number += 1
            return True
        return False

    def kafl_aggregate(self, models, client_ids):
        sorted_list = sorted([(index, value) for index, value in enumerate(self.commit_num)], key=lambda x: x[1])
        sorted_indices = sorted(range(len(self.commit_num)), key=lambda k: self.commit_num[k])
        result = [sorted_indices.index(i) for i in range(len(self.commit_num))]
        q_list = [sorted_list[98 + 1 - result[id]][1] / sum(self.commit_num) for id in client_ids]
        p_list=[]
        count=0
        for cid in client_ids:
            p_list.append(abs(self.clients[cid].datavol) * math.exp(-self.r * abs(self.lambada[cid] / q_list[count])))
            count=count+1

        weight_list = [pi / sum(p_list) for pi in p_list]
        weighted_models = [weight * model for weight, model in zip(weight_list, models)]
        w_new = fmodule._model_sum(weighted_models)
        return (1 - self.alpha) * self.model + self.alpha * w_new

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

    def compute_lambda(self, w_local):
        w_global = self.model
        l2_norm_squared = self.compute_l2_difference(w_global, w_local)

        return l2_norm_squared

    def compute_l2_difference(self, model1, model2):
        # 获取模型的所有参数
        params1 = [p.data for p in model1.parameters()]
        params2 = [p.data for p in model2.parameters()]

        # 用来存储总的 L2 范数差异
        total_l2_diff = 0.0
        fenzi = 0.0
        # 对每一层的权重和偏置计算 L2 范数差异
        for p1, p2 in zip(params1, params2):
            # 计算 L2 范数差异并加到总差异中
            l2_diff = torch.norm(p2, 2)
            total_l2_diff += l2_diff.item() ** 2  # L2范数的平方和
            fenzi += torch.sum(p1 * p2)
        # 取平方根得到最终的 L2 范数差异

        return fenzi / total_l2_diff - 1


class Client(BasicClient):
    def __init__(self, option={}):
        super().__init__(option)
        self.mu = None

    def initialize(self, *args, **kwargs):
        self.actions = {0: self.reply, 1: self.send_model, 2: self.rp,3:self.receive_model}
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
