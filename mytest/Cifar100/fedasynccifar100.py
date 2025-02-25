import flgo
import flgo.algorithm.fedasync_own as  fedasync
import flgo.benchmark.mnist_classification as mnist
import flgo.benchmark.cifar100_classification as cifar100
import flgo.benchmark.partition as fbp

task = './cifar100_experiment' # task name
flgo.gen_task_by_(cifar100, fbp.DirichletPartitioner(num_clients=100, alpha=0.011), task)
#100选20客户端

if __name__ == '__main__':

    fedavg_runner = flgo.init(task=task, algorithm=fedasync, option={'num_rounds': 500, 'num_epochs': 100, 'gpu': 0})
    fedavg_runner.run()
