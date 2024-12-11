"""
This is a non-official implementation of 'Federated Optimization in Heterogeneous
Networks' (http://arxiv.org/abs/1812.06127)
"""
from flgo.algorithm.fedbase import BasicServer, BasicClient
import copy
import torch
from flgo.utils import fmodule

class Server(BasicServer):
    def initialize(self, *args, **kwargs):
        self.init_algo_para({'mu':0.1})
        self.sample_option = 'md'
        self.aggregation_option = 'uniform'

class Client(BasicClient):
    @fmodule.with_multi_gpus
    def train(self, model):
        # global parameters
        '''
        改动，
        原：对服务器下发的模型进行训练，返回给服务器
        新：对本地模型进行训练，返回给服务器，将本地模型赋值为全局模型
        '''
        if self.model!=None:
            train_model = self.model

        else : train_model = model
        src_model = copy.deepcopy(train_model)
        src_model.freeze_grad()
        model.train()
        optimizer = self.calculator.get_optimizer(train_model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in range(self.num_steps):
            # get a batch of data
            batch_data = self.get_batch_data()
            train_model.zero_grad()
            # calculate the loss of the model on batched dataset through task-specified calculator
            loss = self.calculator.compute_loss(train_model, batch_data)['loss']
            loss_proximal = 0
            for pm, ps in zip(model.parameters(), src_model.parameters()):
                loss_proximal += torch.sum(torch.pow(pm - ps, 2))
            loss = loss + 0.5 * self.mu * loss_proximal
            loss.backward()
            optimizer.step()
        self.model = model
        return