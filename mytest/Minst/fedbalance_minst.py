import flgo
import flgo.benchmark.mnist_classification as mnist
import flgo.benchmark.partition as fbp
import flgo.algorithm.fedbalance as fedbalance

import flgo.experiment.logger.test_logger as testlogger
from flgo.experiment.logger.fedbalance_logger import FedBalanceLogger as logger

task = './mnist_experiment'  # task name
flgo.gen_task_by_(mnist, fbp.DirichletPartitioner(num_clients=100, alpha=0.01), task)

if __name__ == '__main__':
    fedavg_runner = flgo.init(task=task, algorithm=fedbalance, Logger=logger,
                              option={'num_rounds': 500,
                                      'num_epochs': 20,
                                      'gpu': 0,
                                      'num_steps': 10,
                                      'alpha': 0.6,
                                      'agg_num': 14,
                                      'fs_index': 50,
                                      'hl_index': 85,
                                      'log_file': True,
                                      'helpLen': 0})
    fedavg_runner.run()
    '''
        self.alpha = 0.6
        self.agg_num = 15
        self.fs_index = 30
        self.hl_index = 80
    '''
