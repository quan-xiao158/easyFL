import flgo
import flgo.algorithm.fedasync_own as  fedasync
import flgo.benchmark.mnist_classification as mnist
import flgo.benchmark.partition as fbp

task = './fedasync' # task name
flgo.gen_task_by_(mnist, fbp.DirichletPartitioner(num_clients=100, alpha=0.01), task)
#100选20客户端

if __name__ == '__main__':

    fedavg_runner = flgo.init(task=task, algorithm=fedasync, option={'num_rounds': 400, 'num_epochs': 20, 'gpu': 0})
    fedavg_runner.run()
