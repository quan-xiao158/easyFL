import flgo
import flgo.algorithm.fedasync_own as fedavg
import flgo.benchmark.mnist_classification as mnist
import flgo.benchmark.partition as fbp
import flgo.experiment.logger.test_logger
from flgo.experiment.logger.test_logger import TestLogger

task = './mnist_experiment' # task name
flgo.gen_task_by_(mnist, fbp.DirichletPartitioner(num_clients=100, alpha=0.01), task)
#100选20客户端

if __name__ == '__main__':

    fedavg_runner = flgo.init(task=task, algorithm=fedavg,Logger=TestLogger, option={'num_rounds': 500, 'num_epochs': 20, 'gpu': 0,"proportion":0.1})
    fedavg_runner.run()