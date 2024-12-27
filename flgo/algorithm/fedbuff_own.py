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
        self.client_fs_list = [-1] * 100  # 服务器记录客户端时间差列表
        self.client_lh_list = [-1] * 100  # 服务器记录客户端上传模型陈旧度列表
        self.decay_list = [0] * 100
        self.round_number = 0
        self.buff_queue = deque()
        '''
        fedbalance  超参数：

        '''
        self.alpha = 0.72
        self.agg_num = 15
    def initialize(self):
        self.init_algo_para({'buffer_ratio': 0.1, 'eta': 1.0})
        self.buffer = []

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

        self.selected_clients.append(self.selected_client)
        self.concurrent_clients.add(self.selected_client)
        received_packages = self.communicate([self.selected_client], None, 2)
        for index in received_packages:
            self.buff_queue.append(index['model'])
        if len(self.buff_queue) == self.buff_len:
            buff_list = []
            for i in range(self.buff_len):
                m=self.buff_queue.pop()
                buff_list.append((m,m._round))
            self.model = self.fedbuff_aggregate(buff_list)
            self.model._round = self.current_round  # 将模型的 _round 属性设置为当前的回合数
            self.communicate(self.selected_clients, self.model, 1)
            self.selected_clients.clear()
            self.concurrent_clients.clear()
            self.round_number += 1
            return True
        return False





    def communicate(self, client_id_list, send_model, mtype):
        self.total_traffic += len(client_id_list)
        model_list = []
        for client_id in client_id_list:
            server_pkg = self.fedbalance_pack_model(send_model)  # 全局模型
            server_pkg['__mtype__'] = mtype
            client_model = self.communicate_with(client_id, package=server_pkg)  # 与客户端进行通信，下发全局模型到本机训练，获取客户端上的模型
            if mtype == 2:
                model_list.append(client_model)
        if mtype == 2:
            return model_list

    def fedbalance_pack_model(self, send_model):
        return {
            "model": copy.deepcopy(send_model),
        }

    def fedbuff_aggregate(self, buffer):
        taus_bf = [b[1] for b in buffer]
        updates_bf = [b[0] for b in buffer]
        weights_bf = [(1 + self.current_round - ctau) ** (-0.5) for ctau in taus_bf]
        model_delta = fmodule._model_average(updates_bf, weights_bf) / len(buffer)
        model = self.model + self.eta * model_delta
        return model


class Client(BasicClient):
    def __init__(self, option={}):
        super().__init__(option)
        self.mu = None

    def initialize(self, *args, **kwargs):
        self.actions = {0: self.reply, 1: self.send_model, 2: self.rp}
        self.mu = 0.1

    def send_model(self, servermodel):
        model = self.unpack(servermodel)
        self.model = model

    def rp(self, servermodel):
        global_model = copy.deepcopy(self.model)
        self.model_train(self.model)
        update = self.model-global_model
        update._round = self.model._round
        cpkg = self.pack(update)
        return cpkg

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

