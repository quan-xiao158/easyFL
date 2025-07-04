import flgo
import flgo.algorithm.fedasync as  fedasync
import flgo.algorithm.fedbuff as fedbuff
import flgo.algorithm.KAFL as KAFL
import flgo.algorithm.fedbalance as fedbalance
import flgo.benchmark.fashion_classification as fashion
import flgo.benchmark.partition as fbp
from flgo.experiment.logger.fedbalance_logger import FedBalanceLogger as logger
task = '../fashion_experiment' # task name
flgo.gen_task_by_(fashion, fbp.DirichletPartitioner(num_clients=100, alpha=0.001), task)
if __name__ == '__main__':
    #已完成
    task1 = flgo.init(task=task, algorithm=fedbalance,Logger=logger, option={'num_rounds': 1000, 'num_epochs': 20, 'gpu': 0,'b':0.3,'t':10,"alpha":0.6,"agg_num":15,"plot":True,"fs_index":50,"hl_index":70,"client_weight":"uniform"})
    task1.run()
