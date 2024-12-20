import flgo
import flgo.algorithm.fedavg as fedavg
import flgo.benchmark.cifar10_classification as cifar10
import flgo.algorithm.fedasync as  fedasync
import flgo.algorithm.fedbalence as fedbalence
import flgo.algorithm.fedbuff as fedbuff
import flgo.benchmark.mnist_classification as mnist
import flgo.benchmark.partition as fbp
import flgo.experiment.analyzer as al

task = './fedasync' # task name
flgo.gen_task_by_(mnist, fbp.DirichletPartitioner(num_clients=100, alpha=0.01), task)
#100选20客户端

if __name__ == '__main__':

    fedavg_runner = flgo.init(task=task, algorithm=fedbalence, option={'num_rounds': 253, 'num_epochs': 20, 'gpu': 0, 'num_steps': 10})
    fedavg_runner.run()
