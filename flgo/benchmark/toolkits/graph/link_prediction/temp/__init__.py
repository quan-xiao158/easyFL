r"""
This module is a template for benchmark of link prediction within a graph in horizontalFL. To use this module,
one can write codes as below

Example::

"""
from .model import default_model
import flgo.benchmark.toolkits.visualization
import flgo.benchmark.toolkits.partition

default_model = default_model
default_partitioner = flgo.benchmark.toolkits.partition.NodeLouvainPartitioner
default_partition_para = {'num_clients':10}
visualize = flgo.benchmark.toolkits.visualization.visualize_by_community

