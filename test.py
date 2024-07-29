import flgo.algorithm.fedavg as fedavg
import flgo.experiment.analyzer
import os

#%%
task = './test_mnist'
gen_config = {'benchmark':{'name':'flgo.benchmark.mnist_classification'},'partitioner':{'name': 'IIDPartitioner','para':{'num_clients':100}}}

if __name__ == '__main__':
    # generate federated task if task doesn't exist
    if not os.path.exists(task): flgo.gen_task(gen_config, task_path=task)
    # running fedavg on the specified task
    # runner = flgo.init(task, fedavg, {'cpu':[0,],'log_file':True, 'num_epochs':1})
    # runner.run()
    fedavg_runner = flgo.init(task=task, algorithm=fedavg, option={'num_rounds': 5, 'num_epochs': 1, 'cpu': 0})
    fedavg_runner.run()
    # visualize the experimental result
    # flgo.experiment.analyzer.show(analysis_plan)
    import flgo.experiment.analyzer as al

    analysis_plan = {
        'Selector': {
            'task': task,
            'header': ['fedavg']
        },
        'Painter': {
            'Curve': [
                {'args': {'x': 'communication_round', 'y': 'val_loss'}, 'fig_option': {'title': 'valid loss on MNIST'}},
                {'args': {'x': 'communication_round', 'y': 'val_accuracy'},
                 'fig_option': {'title': 'valid accuracy on MNIST'}},
            ]
        }
    }
    al.show(analysis_plan)