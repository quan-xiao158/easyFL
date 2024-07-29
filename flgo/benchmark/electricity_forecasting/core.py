from flgo.benchmark.toolkits.series.forecasting import BuiltinClassGenerator, BuiltinClassPipe, GeneralCalculator
import os
import flgo
from flgo.benchmark.toolkits.series.forecasting.datasets import Electricity

path = os.path.join(flgo.benchmark.data_root, 'ELECTRICITY')
builtin_class = Electricity

class TaskGenerator(BuiltinClassGenerator):
    def __init__(self, rawdata_path=path):
        super(TaskGenerator, self).__init__(os.path.split(os.path.dirname(__file__))[-1], rawdata_path, builtin_class)

class TaskPipe(BuiltinClassPipe):
    def __init__(self, task_path):
        super(TaskPipe, self).__init__(task_path, builtin_class)

TaskCalculator = GeneralCalculator
