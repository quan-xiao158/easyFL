import flgo
import flgo.algorithm.fedasync as  fedasync
import flgo.algorithm.fedavg as fedavg
import flgo.algorithm.ca2fl as ca2fl
import flgo.algorithm.fedbalance as fedbalance
import flgo.algorithm.fedbuff as fedbuff
import flgo.benchmark.mnist_classification as mnist
import flgo.benchmark.partition as fbp
from flgo.experiment.logger.fedbalance_logger import FedBalanceLogger as logger

task = './mnist_experiment' # task name
flgo.gen_task_by_(mnist, fbp.DirichletPartitioner(num_clients=100, alpha=0.01), task)
#100选20客户端

if __name__ == '__main__':

    task4 = flgo.init(task=task, algorithm=fedbalance,Logger=logger, option={'num_rounds': 500, 'num_epochs': 20,"proportion":0.1,'gpu': 0,'b':0.7,'t':1,"plot":True,"client_weight":"uniform"})
    task4.run()

