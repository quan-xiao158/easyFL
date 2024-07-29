import flgo
import flgo.algorithm.fedavg as fedavg
import flgo.benchmark.cifar10_classification as cifar10
import flgo.benchmark.partition as fbp

task = './dir0.3_cifar' # task name
flgo.gen_task_by_(cifar10, fbp.DirichletPartitioner(num_clients=100, alpha=0.3), 'dir0.3_cifar')
#100选20客户端

if __name__ == '__main__':

    fedavg_runner = flgo.init(task=task, algorithm=fedavg, option={'num_rounds': 100, 'num_epochs': 4, 'cpu': 0})
    fedavg_runner.run()

    import flgo.experiment.analyzer as al

    analysis_plan = {
        'Selector': {
            'task': task,
            'header': ['fedavg']
        },
        'Painter': {
            'Curve': [
                {'args': {'x': 'communication_round', 'y': 'val_loss'}, 'fig_option': {'title': 'valid loss on CRF10'}},
                {'args': {'x': 'communication_round', 'y': 'val_accuracy'},
                 'fig_option': {'title': 'valid accuracy on MNIST'}},
            ]
        }
    }
    al.show(analysis_plan)