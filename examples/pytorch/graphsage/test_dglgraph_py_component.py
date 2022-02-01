import dgl
import numpy as np
# import torch as th
import time
import argparse
import tqdm

from load_graph import  load_ogb

print('importing')
# test
import test_dglgraph_cpp_component

# load and create a DGLGraph in python

print('loading')
g, n_classes = load_ogb('ogbn-products')
print(f'graph type: {type(g)}')
print(f'graph attributes {g.__dict__}')
print(f'_graph attributes {g._graph.__dict__}')
train_g = val_g = test_g = g
train_nfeat = val_nfeat = test_nfeat = g.ndata.pop('features')
train_labels = val_labels = test_labels = g.ndata.pop('labels')
# Pack data
data = n_classes, train_g, val_g, test_g, train_nfeat, train_labels, \
       val_nfeat, val_labels, test_nfeat, test_labels

# call a C++ function that works with the DGLGraph
print('add')
print(test_dglgraph_cpp_component.add(4, 5))
print('accept_graph')
# print(test_dglgraph_cpp_component.accept_graph(1))
print(test_dglgraph_cpp_component.accept_graph(g._graph))

print('Finished')
