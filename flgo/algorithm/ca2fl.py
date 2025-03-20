"""This is a non-official implementation of 'Tackling the Data Heterogeneity in Asynchronous Federated Learning with Cached Update Calibration' (https://openreview.net/forum?id=4aywmeb97I). """
from collections import deque

import torch
from flgo.algorithm.asyncbase import AsyncServer
from flgo.algorithm.fedbase import BasicClient
import flgo.utils.fmodule as fmodule
import copy
class Server(AsyncServer):
    def initialize(self):
        self.init_algo_para({'buffer_ratio': 0.1, 'eta': 1.0})
        self.buffer = []
        self.hs = [torch.tensor(0.) for _ in self.clients]
        self.ht = torch.tensor(0.).to(self.device)
        self.delta = self.model.zeros_like()
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
    def iterate(self):

        self.current_round += 1

        self.selected_client = self.sample_async()
        # 如果有客户端被选中，记录一条日志，显示被选中的客户端及当前时间

        self.selected_clients.append(self.selected_client)
        self.concurrent_clients.add(self.selected_client)
        received_packages = self.communicate([self.selected_client], None, 2)
        for index in received_packages:
            self.buff_queue.append(index['model'])
        if len(self.buff_queue) == self.buff_len:
            self.computeDifference()
            buff_list = []
            for i in range(self.buff_len):
                m=self.buff_queue.pop()
                buff_list.append(m)
            self.model = self.ca2fl_aggregate(buff_list,self.selected_clients)
            self.model._round = self.current_round  # 将模型的 _round 属性设置为当前的回合数
            self.communicate(self.selected_clients, self.model, 1)
            self.selected_clients.clear()
            self.concurrent_clients.clear()
            self.round_number += 1
            return True
        return False
    def package_handler(self, received_packages:dict):
        if self.is_package_empty(received_packages): return False
        received_updates = received_packages['model']
        received_client_ids = received_packages['__cid']
        for cdelta, cid in zip(received_updates, received_client_ids):
            self.delta += (cdelta - self.hs[cid].to(self.device)) if not isinstance(self.hs[cid],torch.Tensor) else cdelta
            self.hs[cid] = cdelta.to('cpu')
            self.buffer.append(cid)
        if len(self.buffer)>= int(self.buffer_ratio * self.num_clients):
            # aggregate and clear updates in buffer
            vt = self.delta / len(self.buffer) + self.ht.to(self.device) if not isinstance(self.ht, torch.Tensor) else self.delta / len(self.buffer)
            self.model = self.model + self.eta * vt
            self.ht = fmodule._model_sum([h for h in self.hs if not isinstance(h, torch.Tensor)]).to(self.device) / self.num_clients
            self.delta = self.model.zeros_like()
            self.buffer = []
            return True
        return False

    def ca2fl_aggregate(self, received_updates,received_client_ids):
        for cdelta, cid in zip(received_updates, received_client_ids):
            self.delta += (cdelta - self.hs[cid].to(self.device)) if not isinstance(self.hs[cid],
                                                                                    torch.Tensor) else cdelta
            self.hs[cid] = cdelta.to('cpu')
            self.buffer.append(cid)

        # aggregate and clear updates in buffer
        vt = self.delta / len(self.buffer) + self.ht.to(self.device) if not isinstance(self.ht,
                                                                                       torch.Tensor) else self.delta / len(
            self.buffer)
        self.model = self.model + self.eta * vt
        self.ht = fmodule._model_sum([h for h in self.hs if not isinstance(h, torch.Tensor)]).to(
            self.device) / self.num_clients
        self.delta = self.model.zeros_like()
        self.buffer = []

        return self.model
    def communicate(self, client_id_list, send_model, mtype):
        self.total_traffic += len(client_id_list)
        model_list = []
        for client_id in client_id_list:
            server_pkg = self.fedbalance_pack_model(send_model)  # 全局模型
            server_pkg['__mtype__'] = mtype
            client_model = self.communicate_with(client_id, package=server_pkg)  # 与客户端进行通信，下发全局模型到本机训练，获取客户端上的模型
            if mtype == 2 or 3:
                model_list.append(client_model)
        if mtype == 2or 3:
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
        self.actions = {0: self.reply, 1: self.send_model, 2: self.rp,3: self.receive_model}
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
