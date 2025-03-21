import flgo
import flgo.algorithm.fedasync as  fedasync
import flgo.algorithm.ca2fl as ca2fl
import  flgo.algorithm.KAFL as Kafl
import flgo.algorithm.fedbuff as fedbuff
import flgo.algorithm.fedbalance as fedbalance
import flgo.benchmark.svhn_classification as svhn
import flgo.benchmark.partition as fbp
from flgo.experiment.logger.fedbalance_logger import FedBalanceLogger as logger
import flgo.benchmark.svhn_classification.model.resnet18_gn as resnet18_gn
task = '../SVHN_experiment' # task name
flgo.gen_task_by_(svhn, fbp.DirichletPartitioner(num_clients=100, alpha=0.011), task)
#100选20客户端

if __name__ == '__main__':
    #Resnet-fedasync
    task1 = flgo.init(task=task,Logger=logger, algorithm=fedasync, option={'num_rounds': 4000, 'num_epochs': 20, 'gpu': 0,'proportion':0.2,'b':0.3,'t':5,"client_weight":"random55",'plot':True},model=resnet18_gn)
    task1.run()
    task2 = flgo.init(task=task,Logger=logger, algorithm=fedasync, option={'num_rounds': 4000, 'num_epochs': 20, 'gpu': 0,'proportion':0.2,'b':0.3,'t':40,"client_weight":"random55",'plot':True},model=resnet18_gn)
    task2.run()
    task3 = flgo.init(task=task, Logger=logger, algorithm=fedasync,option={'num_rounds': 4000, 'num_epochs': 20, 'gpu': 0, 'proportion': 0.2, 'b': 0.3, 't': 100,"client_weight": "random55", 'plot': True}, model=resnet18_gn)
    task3.run()
    task4 = flgo.init(task=task,Logger=logger, algorithm=fedasync, option={'num_rounds': 4000, 'num_epochs': 20, 'gpu': 0,'proportion':0.2,'b':0.7,'t':5,"client_weight":"random55",'plot':True},model=resnet18_gn)
    task4.run()
    task5 = flgo.init(task=task, Logger=logger, algorithm=fedasync,option={'num_rounds': 4000, 'num_epochs': 20, 'gpu': 0, 'proportion': 0.2, 'b': 0.7, 't': 40,"client_weight": "random55", 'plot': True}, model=resnet18_gn)
    task5.run()
    task6 = flgo.init(task=task,Logger=logger, algorithm=fedasync, option={'num_rounds': 4000, 'num_epochs': 20, 'gpu': 0,'proportion':0.2,'b':0.7,'t':100,"client_weight":"random55",'plot':True},model=resnet18_gn)
    task6.run()
#Resnet-fedbuff
    task1 = flgo.init(task=task,Logger=logger, algorithm=fedbuff, option={'num_rounds': 4000, 'num_epochs': 20, 'gpu': 0,'proportion':0.2,'b':0.3,'t':5,"client_weight":"random55",'plot':True},model=resnet18_gn)
    task1.run()
    task2 = flgo.init(task=task,Logger=logger, algorithm=fedbuff, option={'num_rounds': 4000, 'num_epochs': 20, 'gpu': 0,'proportion':0.2,'b':0.3,'t':40,"client_weight":"random55",'plot':True},model=resnet18_gn)
    task2.run()
    task3 = flgo.init(task=task, Logger=logger, algorithm=fedbuff,option={'num_rounds': 4000, 'num_epochs': 20, 'gpu': 0, 'proportion': 0.2, 'b': 0.3, 't': 100,"client_weight": "random55", 'plot': True}, model=resnet18_gn)
    task3.run()
    task5 = flgo.init(task=task,Logger=logger, algorithm=fedbuff, option={'num_rounds': 4000, 'num_epochs': 20, 'gpu': 0,'proportion':0.2,'b':0.7,'t':5,"client_weight":"random55",'plot':True},model=resnet18_gn)
    task5.run()
    task6 = flgo.init(task=task, Logger=logger, algorithm=fedbuff,option={'num_rounds': 4000, 'num_epochs': 20, 'gpu': 0, 'proportion': 0.2, 'b': 0.7, 't': 40,"client_weight": "random55", 'plot': True}, model=resnet18_gn)
    task6.run()
    task7 = flgo.init(task=task,Logger=logger, algorithm=fedbuff, option={'num_rounds': 4000, 'num_epochs': 20, 'gpu': 0,'proportion':0.2,'b':0.7,'t':100,"client_weight":"random55",'plot':True},model=resnet18_gn)
    task7.run()
#Resnet-KAFL
    task1 = flgo.init(task=task,Logger=logger, algorithm=Kafl, option={'num_rounds': 4000, 'num_epochs': 20, 'gpu': 0,'proportion':0.2,'b':0.3,'t':5,"client_weight":"random55",'plot':True},model=resnet18_gn)
    task1.run()
    task2 = flgo.init(task=task,Logger=logger, algorithm=Kafl, option={'num_rounds': 4000, 'num_epochs': 20, 'gpu': 0,'proportion':0.2,'b':0.3,'t':40,"client_weight":"random55",'plot':True},model=resnet18_gn)
    task2.run()
    task3 = flgo.init(task=task, Logger=logger, algorithm=Kafl,option={'num_rounds': 4000, 'num_epochs': 20, 'gpu': 0, 'proportion': 0.2, 'b': 0.3, 't': 100,"client_weight": "random55", 'plot': True}, model=resnet18_gn)
    task3.run()
    task5 = flgo.init(task=task,Logger=logger, algorithm=Kafl, option={'num_rounds': 4000, 'num_epochs': 20, 'gpu': 0,'proportion':0.2,'b':0.7,'t':5,"client_weight":"random55",'plot':True},model=resnet18_gn)
    task5.run()
    task6 = flgo.init(task=task, Logger=logger, algorithm=Kafl,option={'num_rounds': 4000, 'num_epochs': 20, 'gpu': 0, 'proportion': 0.2, 'b': 0.7, 't': 40,"client_weight": "random55", 'plot': True}, model=resnet18_gn)
    task6.run()
    task7 = flgo.init(task=task,Logger=logger, algorithm=Kafl, option={'num_rounds': 4000, 'num_epochs': 20, 'gpu': 0,'proportion':0.2,'b':0.7,'t':100,"client_weight":"random55",'plot':True},model=resnet18_gn)
    task7.run()
#Resnet-Fedbalance
    task10 = flgo.init(task=task, algorithm=fedbalance, Logger=logger,option={ 'b': 0.3, 't': 5,'fs_index': 50,'hl_index': 85,'num_rounds': 7000, 'num_epochs': 20,'alpha': 0.6,'agg_num': 14,'gpu': 0, "plot": True,"client_weight": "uniform"},model=resnet18_gn)
    task10.run()
    task11 = flgo.init(task=task, algorithm=fedbalance, Logger=logger,option={ 'b': 0.3, 't': 40,'fs_index': 50,'hl_index': 85,'num_rounds': 7000, 'num_epochs': 20,'alpha': 0.6,'agg_num': 14,'gpu': 0, "plot": True,"client_weight": "uniform"},model=resnet18_gn)
    task11.run()
    task12 = flgo.init(task=task, algorithm=fedbalance, Logger=logger,option={ 'b': 0.3, 't': 100,'fs_index': 50,'hl_index': 85,'num_rounds': 7000, 'num_epochs': 20,'alpha': 0.6,'agg_num': 14,'gpu': 0, "plot": True,"client_weight": "uniform"},model=resnet18_gn)
    task12.run()
    task13 = flgo.init(task=task, algorithm=fedbalance, Logger=logger,option={ 'b': 0.7, 't': 5,'fs_index': 50,'hl_index': 85,'num_rounds': 7000, 'num_epochs': 20,'alpha': 0.6,'agg_num': 14,'gpu': 0, "plot": True,"client_weight": "uniform"},model=resnet18_gn)
    task13.run()
    task14 = flgo.init(task=task, algorithm=fedbalance, Logger=logger,option={ 'b': 0.7, 't': 40,'fs_index': 50,'hl_index': 85,'num_rounds': 7000, 'num_epochs': 20,'alpha': 0.6,'agg_num': 14,'gpu': 0, "plot": True,"client_weight": "uniform"},model=resnet18_gn)
    task14.run()
    task15 = flgo.init(task=task, algorithm=fedbalance, Logger=logger,option={ 'b': 0.7, 't': 100,'fs_index': 50,'hl_index': 85,'num_rounds': 7000, 'num_epochs': 20,'alpha': 0.6,'agg_num': 14,'gpu': 0, "plot": True,"client_weight": "uniform"},model=resnet18_gn)
    task15.run()

