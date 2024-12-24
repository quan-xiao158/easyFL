import flgo
import flgo.benchmark.mnist_classification as mnist
import flgo.benchmark.partition as fbp
import flgo.algorithm.fedbalence as fedbalence

task = './mnist_experiment'  # task name
flgo.gen_task_by_(mnist, fbp.DirichletPartitioner(num_clients=100, alpha=0.01), task)
#100选20客户端

if __name__ == '__main__':
    fedavg_runner = flgo.init(task=task, algorithm=fedbalence,
                              option={'num_rounds': 500,
                                      'num_epochs': 20,
                                      'gpu': 0,
                                      'num_steps': 10,
                                      'alpha': 0.3,
                                      'agg_num': 15,
                                      'fs_index': 65,
                                      'hl_index': 65})
    fedavg_runner.run()
    '''
        self.alpha = 0.3
        self.agg_num = 15
        self.fs_index = 65
        self.hl_index = 65
    '''
