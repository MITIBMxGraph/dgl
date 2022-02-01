import dgl
import torch

block1 = dgl.create_block(([0, 1, 2], [1, 2, 3]), num_src_nodes=3, num_dst_nodes=4)
print('block1')
print(block1)
print(block1.__dict__)
print(block1.adj_sparse('coo'))
block2 = dgl.create_block(('csr', ([0, 1, 2, 3], [1, 2, 3], [])), num_src_nodes=3, num_dst_nodes=4)
print('block2')
print(block2)
print(block2.__dict__)
print(block2.adj_sparse('coo'))
