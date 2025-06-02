import flgo
import flgo.algorithm.fedasync as  fedasync
import flgo.algorithm.fedbuff as fedbuff
import flgo.algorithm.KAFL as KAFL
import flgo.algorithm.fedbalance as fedbalance
import flgo.benchmark.cifar100_classification as cf
import flgo.benchmark.partition as fbp
from flgo.experiment.logger.fedbalance_logger import FedBalanceLogger as logger
task = './cifar100-10' # task name
flgo.gen_task_by_(cf, fbp.DirichletPartitioner(num_clients=100, alpha=10), task)

