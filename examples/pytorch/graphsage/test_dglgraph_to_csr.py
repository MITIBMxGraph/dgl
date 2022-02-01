import torch
import dgl
import numpy as np
import time
import argparse
import tqdm

from load_graph import  load_ogb

# should be able to use torch_sparse
from torch_sparse import SparseTensor

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

#hetero_graph_index = g._graph
#print(g.etypes)
# WARNING: assuming only 1 canonical_etype
# below looks to work but cols are not sorted within the row
#nrows, ncols, indptr, indices, data = hetero_graph_index.adjacency_matrix_tensors(g._etypes_invmap[g._canonical_etypes[0]], False, 'csr')

# instead of above
# cleaner to call adj_sparse
rowptr, col, edge_ids = g.adj_sparse('csr')

#print('nrows')
#print(nrows)
#print('ncols')
#print(ncols)
print('rowptr')
print(len(rowptr))
print(rowptr)
print('col')
print(len(col))
#torch.set_printoptions(edgeitems=174)
print(col)
#torch.set_printoptions(profile='default')
print('edge_ids')
print(len(edge_ids))
print(edge_ids)

# load_ogb function substitutes list of train indices for mask, undo this
train_mask = g.ndata.pop('train_mask')
train_nid = [i for i in range(len(train_mask)) if train_mask[i]]
print('train_nid')
#print(train_nid)

"""
# takes too long
# but by converting to coo and then using SparseTensor can get exactly what Salient normally uses
# thing is adj_t.to_symmetric().csr() takes forever I believe (either that or creating the SparseTensor)

nrows, ncols, row, col = hetero_graph_index.adjacency_matrix_tensors(g._etypes_invmap[g._canonical_etypes[0]], False, 'coo')
print('nrows')
print(nrows)
print('ncols')
print(ncols)
print('row')
print(len(row))
print(row)
print('col')
print(len(col))
print(col)


num_nodes = train_nfeat.size(0)
# trying with torch_sparse
#adj_t = SparseTensor(row=indptr, col=indices, value=None,
#                     sparse_sizes=(num_nodes, num_nodes)).t()
adj_t = SparseTensor(row=row, col=col, value=None,
                     sparse_sizes=(num_nodes, num_nodes)).t()
print('Sparse Tensor created, converting to_symmetric and then to csr')
rowptr, col, _ = adj_t.to_symmetric().csr()
print('rowptr')
print(len(rowptr))
print(rowptr)
print('col')
print(len(col))
print(col)
"""

print('Finished')
