import flgo
import flgo.algorithm.fedasync as  fedasync
import flgo.benchmark.mnist_classification as mnist
import flgo.benchmark.partition as fbp
from flgo.experiment.logger.fedbalance_logger import FedBalanceLogger as logger

task = './mnist_experiment' # task name
flgo.gen_task_by_(mnist, fbp.DirichletPartitioner(num_clients=100, alpha=0.01), task)
#100选20客户端

if __name__ == '__main__':

    fedavgrunner = flgo.init(task=task, algorithm=fedasync,Logger=logger, option={'num_rounds': 1000, 'num_epochs': 20, 'gpu': 0,'b':0.7,'t':1,'plot':False,'client_weight':"uniform"})
    fedavgrunner.run()
    #todo 将round改为1000 效果不明显就0和无穷大
