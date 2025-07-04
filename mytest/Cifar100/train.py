import flgo.benchmark.cifar10_classification.model.cnn as cnn
import flgo
import flgo.algorithm.fedasync as  fedasync
import  flgo.algorithm.fedavg as fedavg
import flgo.benchmark.mnist_classification as mnist
import flgo.algorithm.ca2fl as ca2fl
import flgo.benchmark.cifar100_classification as cifar100
import flgo.benchmark.partition as fbp
from flgo.experiment.logger.fedbalance_logger import FedBalanceLogger as logger
from flgo.experiment.logger.test_logger import TestLogger as TestLogger
import flgo.benchmark.cifar100_classification.model.resnet18_gn as resnet18_gn

task = './cifar100_experiment' # task name
flgo.gen_task_by_(cifar100, fbp.DirichletPartitioner(num_clients=100, alpha=10), task)
#100选20客户端

if __name__ == '__main__':

    task1 = flgo.init(task=task,Logger=TestLogger, algorithm=fedasync, option={'num_rounds': 4000, 'num_epochs': 20, 'gpu': 0,'b':0.7,'t':5,"client_weight":"random55",'plot':True},model=resnet18_gn)
    task1.run()


