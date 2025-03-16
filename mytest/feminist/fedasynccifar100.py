import flgo
import flgo.algorithm.fedavg_own as  fedasync
import flgo.benchmark.mnist_classification as mnist
import flgo.benchmark.fashion_classification as fashion
import flgo.benchmark.partition as fbp
from flgo.experiment.logger.test_logger import TestLogger

task = './fashion_experiment' # task name
flgo.gen_task_by_(fashion, fbp.DirichletPartitioner(num_clients=100, alpha=0.011), task)
#100选20客户端

if __name__ == '__main__':

    fedavg_runner = flgo.init(task=task, Logger=TestLogger,algorithm=fedasync, option={'num_rounds': 500, 'num_epochs': 100, 'cpu': 0})
    fedavg_runner.run()
