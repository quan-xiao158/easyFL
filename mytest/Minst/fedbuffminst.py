import flgo
import flgo.algorithm.fedbuff_own as  fedbuff
import flgo.benchmark.mnist_classification as mnist
import flgo.benchmark.partition as fbp
from flgo.experiment.logger.test_logger import TestLogger
from flgo.experiment.logger.fedbalance_logger import FedBalanceLogger as logger
task = './mnist_experiment' # task name
flgo.gen_task_by_(mnist, fbp.DirichletPartitioner(num_clients=100, alpha=0.01), task)
#100选20客户端

if __name__ == '__main__':

    fedavg_runner = flgo.init(task=task, algorithm=fedbuff, Logger=logger,option={'num_rounds': 500, 'num_epochs': 20, 'gpu': 0,'b':0.7,'t':40})
    fedavg_runner.run()
