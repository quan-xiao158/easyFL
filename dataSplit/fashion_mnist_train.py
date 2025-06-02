import flgo
import flgo.algorithm.fedasync as  fedasync
import flgo.algorithm.fedbuff as fedbuff
import flgo.algorithm.KAFL as KAFL
import flgo.algorithm.fedbalance as fedbalance
import flgo.benchmark.cifar100_classification as fashion
import flgo.benchmark.partition as fbp
from flgo.experiment.logger.fedbalance_logger import FedBalanceLogger as logger
task = './CIFAR100_experiment' # task name
flgo.gen_task_by_(fashion, fbp.DirichletPartitioner(num_clients=100, alpha=1), task)
