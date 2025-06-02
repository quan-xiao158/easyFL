import flgo.benchmark.cifar10_classification.model.cnn as cnn
import flgo
import flgo.algorithm.fedasync as  fedasync
import flgo.algorithm.fedbuff as fedbuff
import  flgo.algorithm.fedavg as fedavg
import flgo.algorithm.KAFL as Kafl
import flgo.algorithm.fedbalance as fedbalance
import flgo.benchmark.mnist_classification as mnist
import flgo.algorithm.ca2fl as ca2fl
import flgo.benchmark.cifar10_classification as cifar10
import flgo.benchmark.partition as fbp
from flgo.experiment.logger.fedbalance_logger import FedBalanceLogger as logger
import flgo.benchmark.cifar10_classification.model.cnn as cnn
import flgo.benchmark.cifar10_classification.model.resnet18_gn as resnet18_gn
task = '../cifar10_experiment' # task name
flgo.gen_task_by_(cifar10, fbp.DirichletPartitioner(num_clients=100, alpha=0.011), task)
#100选20客户端
import torch
if __name__ == '__main__':
# 3500轮后现存不够
    task1 = flgo.init(task=task, algorithm=fedasync,Logger=logger, option={'num_rounds': 3000, 'num_epochs': 20, 'gpu': 0,'b':0.3,'t':10,"plot":True,"client_weight":"uniform"})
    task1.run()
    task2 = flgo.init(task=task, algorithm=fedasync, Logger=logger,option={'num_rounds': 3000, 'num_epochs': 20, 'gpu': 0, 'b': 0.3, 't': 40, "plot": True,"client_weight": "uniform"})
    task2.run()


