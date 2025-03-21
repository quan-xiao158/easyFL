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
        src_model = copy.deepcopy(model)
        src_model.freeze_grad()
        model.train()
        optimizer = self.calculator.get_optimizer(model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in range(self.num_epochs):
            # get a batch of data
            batch_data = self.get_batch_data()
            model.zero_grad()
            # calculate the loss of the model on batched dataset through task-specified calculator
            loss = self.calculator.compute_loss(model, batch_data)['loss']
            loss_proximal = 0
            for pm, ps in zip(model.parameters(), src_model.parameters()):
                loss_proximal += torch.sum(torch.pow(pm - ps, 2))
            loss = loss + 0.5 * self.mu * loss_proximal
            loss.backward()
            if self.clip_grad>0:torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.clip_grad)
            optimizer.step()
        return