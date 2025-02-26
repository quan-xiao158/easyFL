import flgo.benchmark.cifar10_classification.model.cnn as cnn
import flgo
import flgo.algorithm.fedbuff_own as  fedbuff
import  flgo.algorithm.fedavg as fedavg
import flgo.benchmark.mnist_classification as mnist
import flgo.benchmark.cifar10_classification as cifar10
import flgo.benchmark.partition as fbp
from flgo.experiment.logger.test_logger import TestLogger

task = './cifar10_experiment' # task name
flgo.gen_task_by_(cifar10, fbp.DirichletPartitioner(num_clients=100, alpha=0.011), task)
#100选20客户端

if __name__ == '__main__':

    fedavg_runner = flgo.init(task=task,Logger=TestLogger, algorithm=fedbuff, option={'num_rounds': 2500, 'num_epochs': 20, 'cpu': 0,'proportion':0.2},model=cnn)
    fedavg_runner.run()
