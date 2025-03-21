import flgo
import flgo.algorithm.fedasync as  fedasync
import flgo.algorithm.ca2fl as ca2fl
import flgo.algorithm.fedbalance as fedbalance
import flgo.benchmark.mnist_classification as mnist
import flgo.benchmark.partition as fbp
from flgo.experiment.logger.fedbalance_logger import FedBalanceLogger as logger

task = './mnist_experiment' # task name
flgo.gen_task_by_(mnist, fbp.DirichletPartitioner(num_clients=100, alpha=0.01), task)
#100选20客户端

if __name__ == '__main__':

    task13 = flgo.init(task=task, algorithm=fedbalance, Logger=logger,option={ 'b': 0.3, 't': 5,'fs_index': 55,'hl_index': 75,'num_rounds': 1000, 'num_epochs': 20,'alpha': 0.6,'agg_num': 14,'gpu': 0, "plot": True,"client_weight": "uniform"})
    task13.run()
    #todo 将round改为1000 效果不明显就0和无穷大
