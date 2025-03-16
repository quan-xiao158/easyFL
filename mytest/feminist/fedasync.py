import flgo
import flgo.algorithm.fedasync as  fedasync
import flgo.benchmark.mnist_classification as mnist
import flgo.benchmark.fashion_classification as fashion
import flgo.benchmark.partition as fbp
from flgo.experiment.logger.fedbalance_logger import FedBalanceLogger as logger

task = './fashion_experiment' # task name
flgo.gen_task_by_(fashion, fbp.DirichletPartitioner(num_clients=100, alpha=0.011), task)
#100选20客户端

if __name__ == '__main__':

    fedavg_runner = flgo.init(task=task, algorithm=fedasync,Logger=logger, option={'num_rounds': 1000, 'num_epochs': 20, 'gpu': 0,'b':0.7,'t':55,'client_weight':'randoms6'})
    fedavg_runner.run()
