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
    # task1 = flgo.init(task=task, algorithm=fedasync,Logger=logger, option={'num_rounds': 1000, 'num_epochs': 20, 'gpu': 0,'b':0.3,'t':100,"plot":True,"client_weight":"uniform"})
    # task1.run()
    #未完成
    # task2 = flgo.init(task=task, algorithm=fedbuff, Logger=logger,option={'num_rounds': 1000, 'num_epochs': 20, 'gpu': 0, 'b': 0.3, 't': 5, "plot": True,"client_weight": "uniform"})
    # task2.run()
    # task3 = flgo.init(task=task, algorithm=fedbuff, Logger=logger,option={'num_rounds': 1000, 'num_epochs': 20, 'gpu': 0, 'b': 0.3, 't': 100, "plot": True,"client_weight": "uniform"})
    # task3.run()
    task4 = flgo.init(task=task, algorithm=fedbuff,Logger=logger, option={'num_rounds': 1000, 'num_epochs': 20, 'gpu': 0,'b':0.7,'t':5,"plot":True,"client_weight":"uniform"})
    task4.run()
    task5 = flgo.init(task=task, algorithm=fedbuff, Logger=logger,option={'num_rounds': 1000, 'num_epochs': 20, 'gpu': 0, 'b': 0.7, 't': 100, "plot": True,"client_weight": "uniform"})
    task5.run()
    task6 = flgo.init(task=task, algorithm=KAFL, Logger=logger,option={'num_rounds': 1000, 'num_epochs': 20, 'gpu': 0, 'b': 0.3, 't': 5, "plot": True,"client_weight": "uniform"})
    task6.run()
    task7 = flgo.init(task=task, algorithm=KAFL, Logger=logger,option={'num_rounds': 1000, 'num_epochs': 20, 'gpu': 0, 'b': 0.3, 't': 100, "plot": True,"client_weight": "uniform"})
    task7.run()
    task8 = flgo.init(task=task, algorithm=KAFL,Logger=logger, option={'num_rounds': 1000, 'num_epochs': 20, 'gpu': 0,'b':0.7,'t':5,"plot":True,"client_weight":"uniform"})
    task8.run()
    task9 = flgo.init(task=task, algorithm=KAFL, Logger=logger,option={'num_rounds': 1000, 'num_epochs': 20, 'gpu': 0, 'b': 0.7, 't': 100, "plot": True,"client_weight": "uniform"})
    task9.run()
    task10 = flgo.init(task=task, algorithm=fedbalance, Logger=logger,option={'num_rounds': 1000, 'num_epochs': 20, 'gpu': 0, 'b': 0.3, 't': 5, "plot": True,"client_weight": "uniform"})
    task10.run()
    task11 = flgo.init(task=task, algorithm=fedbalance, Logger=logger,option={'num_rounds': 1000, 'num_epochs': 20, 'gpu': 0, 'b': 0.3, 't': 100, "plot": True,"client_weight": "uniform"})
    task11.run()
    task12 = flgo.init(task=task, algorithm=fedbalance,Logger=logger, option={'num_rounds': 1000, 'num_epochs': 20, 'gpu': 0,'b':0.7,'t':5,"plot":True,"client_weight":"uniform"})
    task12.run()
    task13 = flgo.init(task=task, algorithm=fedbalance, Logger=logger,option={'num_rounds': 1000, 'num_epochs': 20, 'gpu': 0, 'b': 0.7, 't': 100, "plot": True,"client_weight": "uniform"})
    task13.run()
