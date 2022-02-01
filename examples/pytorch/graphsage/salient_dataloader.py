import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.nn.pytorch as dglnn
import time
import argparse
import tqdm

from model import SAGE
from load_graph import load_reddit, inductive_split, load_ogb

# fast sampler
from fast_trainer.samplers import *
from fast_trainer.transferers import *

# profiling
import nvtx

@nvtx.annotate('compute_acc')
def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    labels = labels.long()
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)

@nvtx.annotate('evaluate')
def evaluate(model, g, nfeat, labels, val_nid, device):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        pred = model.inference(g, nfeat, device, args.batch_size, args.num_workers)
    model.train()
    return compute_acc(pred[val_nid], labels[val_nid].to(pred.device))


#### Entry point
def run(args, device, data):
    # Unpack data
    #n_classes, train_g, val_g, test_g, train_nfeat, train_labels, \
    #val_nfeat, val_labels, test_nfeat, test_labels = data
    n_classes, g, x, y = data
    x_dim = x.shape[1]
    train_nid = th.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
    val_nid = th.nonzero(g.ndata['val_mask'], as_tuple=True)[0]
    test_nid = th.nonzero(~(g.ndata['train_mask'] | g.ndata['val_mask']), as_tuple=True)[0]

    # if want to be consistent with DGL's reversed notation must reverse fanout
    dgl__fanout = [int(f) for f in args.fan_out.split(',')]
    salient__fanout = list(reversed(dgl__fanout))

    # Create DGL DataLoader (similar to pytorch) for constructing blocks
    """
    dataloader_device = th.device('cpu')
    dgl__sampler = dgl.dataloading.MultiLayerNeighborSampler(fanout)
    dgl__dataloader = dgl.dataloading.NodeDataLoader(
         train_g,
         train_nid,
         dgl__sampler,
         device=dataloader_device,
         batch_size=args.batch_size,
         shuffle=False,                  # disabled for comparison purposes only
         drop_last=False,
         num_workers=args.num_workers)
    """

    #convert graph to input to format that can be fed into Salient
    with nvtx.annotate('convert DGLGraph to CSR for Salient'):
        rowptr, col, edge_ids = g.adj_sparse('csr')
    """
    print('Checking if all nodes have neighbors')
    for i in range(len(rowptr)-1):
        if (rowptr[i] == rowptr[i+1]):
            print(f'{i} has no neighbors')
    coo_row, coo_col = g.adj_sparse('coo')
    print(len(coo_row))
    print(len(coo_col))
    print('catting')
    cat = torch.cat((coo_row, coo_col))
    print('unique')
    print(len(torch.unique(cat)))
    """

    # using torch sparse
    #import torch_sparse
    #pre_row, pre_col = g.adj_sparse('coo') num_nodes = x.size(0)
    #adj_t = SparseTensor(row=pre_row, col=pre_col, value=None,
    #                     sparse_sizes=(num_nodes, num_nodes)).t()
    #rowptr, col, _ = adj_t.to_symmetric().csr()


    # create Salient dataloader
    #print('tensor types')
    #print(x.type())
    #print(x.__dict__)
    #print(y.type())
    cfg = FastSamplerConfig(
                # ignore features and labels temporarily
                #x=dataset.x, y=dataset.y.unsqueeze(-1),
                #x=th.empty((0, 0)), y=th.empty((0, 0)),
                x=x, y=y.unsqueeze(-1),
                rowptr=rowptr, col=col,
                idx=train_nid,
                batch_size=args.batch_size, sizes=salient__fanout,
                #skip_nonfull_batch=False, pin_memory=True
                skip_nonfull_batch=False, pin_memory=True # debug pinned memory
            )

    train_loader = FastSampler(40, 50, cfg)
    #train_loader = FastSampler(1, 50, cfg) # serial to debug

    # Define model and optimizer
    model = SAGE(x_dim, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout)
    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Comparison loop
    iter_tput = []
    for epoch in range(args.num_epochs):
        with nvtx.annotate('epoch'):
            tic = time.time()
            tic_step = time.time()

            # Loop over the dataloader to sample the computation dependency graph as a list of
            # blocks.
            #dgl__dataloader_iter = iter(dgl__dataloader)
            train_loader_iter = iter(train_loader)
            salient__dataloader = DevicePrefetcher([device], train_loader_iter)
            step = -1

            while True:

                with nvtx.annotate('iteration'):
                    step += 1
                    #print(step)

                    """
                    # load DGL prepared batch
                    try:
                       dgl__input_nodes, dgl__seeds, dgl__blocks = next(dgl__dataloader_iter)
                    except StopIteration:
                        # make sure DGL Dataloader and Salient complete simultaneously
                        inputs = next(salient__dataloader, [])
                        assert len(inputs) == 0
                        break

                    """
                    # load Salient prepared batch
                    inputs = next(salient__dataloader, [])
                    #assert len(inputs) != 0:
                    if len(inputs) == 0:
                        break
                    """

                    # add some more assertions on types returned by different dataloaders
                    #print(step)
                    #print(f'type dgl rv: {type(blocks)}')
                    #print(f'type sal rv: {type(inputs)}')
                    #print('blocks')
                    #print(blocks)
                    #print('inputs')
                    #print(inputs)
                    #print()
                    """

                    # inspect returned adj
                    # print(batch.adjs)
                    #print('loop over batch.adjs')
                    #for adj in batch.adjs:
                    #    print(adj.adj_t.csr())

                    # convert Salient MFG to DGL MFG
                    """
                    with nvtx.annotate('convert Salient MFG to DGL MFG', color='cyan'):
                        blocks = []
                        for adj in batch.adjs:
                            with nvtx.annotate('adj loop iteration'):
                                # should just be references, not copying..
                                # need to also do a tranpose here since dgl's MFG has a flipped
                                # notion of direction
                                #with nvtx.annotate('adj transpose + csr'):
                                    #rowptr, col, edge_ids = adj.adj_t.t().csr()
                                #with nvtx.annotate('adj transpose'):
                                #    adj_transpose = adj.adj_t.t()
                                #with nvtx.annotate('adj csr'):
                                #    rowptr, col, edge_ids = adj_transpose.csr()
                               # with nvtx.annotate('csc'):
                               #     rowptr, col, edge_ids = adj.adj_t.csc()
                                # returns edge_ids as None instead of an empty tensor
                                with nvtx.annotate('adj extract csr'):
                                    rowptr, col, edge_ids = adj.adj_t.csr()
                                edge_ids = th.empty(0) if edge_ids is None else edge_ids

                                #print('create_block')
                                #print(adj)
                                #print('rowptr')
                                #print(rowptr)
                                #print('col')
                                #print(col)
                                with nvtx.annotate('create_block from csr'):
                                    block = dgl.create_block(('csr', (rowptr, col, edge_ids)),
                                                             num_src_nodes=adj.size[1],
                                                             num_dst_nodes=adj.size[0])
                                # looks like also need coo to set _ID parameters in blocks
                                # can potentially just optimize to coo
                                #print('block.__dict__')
                                #print(block.__dict__)
                                #print('adj.adj_t.coo()')
                                #print(adj.adj_t.coo())

                                #src_nodes, dst_nodes, _ = adj.adj_t.t().coo()
                                with nvtx.annotate('src_nodes, dst_nodes'):
                                    src_nodes = torch.arange(block.number_of_src_nodes())
                                    dst_nodes = torch.arange(block.number_of_dst_nodes())

                                #print(f'sz src_nodes: {len(torch.unique(src_nodes))}')
                                #print(f'sz dst_nodes: {len(torch.unique(dst_nodes))}')
                                #print(f'len rowptr: {len(rowptr)}')
                                #print(f'len col: {len(col)}')
                                #print(f'len src_nodes: {len(src_nodes)}')
                                #print(f'len dst_nodes: {len(dst_nodes)}')
                                # torch.unique calls seem not efficient at all
                                # dst nodes also need to include src ndoes

                                #dst_nodes = torch.unique(dst_nodes)
                                #src_nodes = torch.unique(torch.cat((src_nodes, dst_nodes)))

                                #print('rowptr')
                                #print(rowptr)
                                #print('col')
                                #print(col)
                                #print('adj')
                                #print(adj)
                                #print('src_nodes')
                                #print(src_nodes)
                                #print('dst_nodes')
                                #print(dst_nodes)
                                #etype = None
                                #etid = block.get_etype_id(etype)
                                #block[etid]['_node_frames'][0]['_ID'] = src_nodes
                                #block[etid]['_node_frames'][1]['_ID'] = dst_nodes
                                #print(f'src_nodes: {torch.sort(src_nodes)}')
                                with nvtx.annotate('_node_frames _ID'):
                                    block._node_frames[0]['_ID'] = src_nodes
                                    block._node_frames[1]['_ID'] = dst_nodes
                                #print(f'dst_nodes: {torch.sort(dst_nodes)}')
                                blocks.append(block)

                        # Identify the input features and output labels
                        # nodes that are considered an 'input' to training are the expanded neighborhood
                        # of the batch nodes
                        # nodes that are considered 'output' nodes are the batch nodes with respect to which
                        # the loss of the minibatch will be calculated

                        # not necessary except for debugging
                        # input_nodes, output_nodes = blocks[0].srcdata[dgl.NID], blocks[-1].dstdata[dgl.NID]
                    # Load the input features as well as output labels
                    # load subtensor is how DGL slices
                    # batch_inputs, batch_labels = load_subtensor(x, y, output_nodes, input_nodes, device)
                    # However salient has already sliced in the c++ threads, so don't do a load_subtensor call
                    #print('batch.x')
                    #print(len(batch.x))
                    #print('batch.y')
                    #print(len(batch.y))

                    # using prefetcher
                    #with nvtx.annotate('transfer x,y to GPU', color='yellow'):
                    #    batch_inputs = batch.x.to(device)
                    #    batch_labels = batch.y.to(device)

                    ## Send everything to device, blocks now points on tensors on device
                    #with nvtx.annotate('transfer MFGs to GPU', color='yellow'):
                    #    blocks = [block.int().to(device) for block in blocks]
                    """

                    for batch in inputs:
                        # Compute loss and prediction
                        with nvtx.annotate('model (async HtoD + forward())'):
                            with th.autograd.profiler.emit_nvtx():
                                #batch_pred = model(blocks, batch_inputs)
                                batch_pred = model(batch.blocks, batch.x)
                        #print('---------- Salient Summary ----------')
                        #print(f'blocks: {blocks}')
                        #print(f'sz input_nodes, sz output_nodes: {len(input_nodes)}, {len(output_nodes)}')
                        #print(f'sz batch.x, sz batch.y: {len(batch.x)}, {len(batch.y)}')
                        #print(f'sz batch_pred: {batch_pred.size()}')
                        #print('---------- --------------- ----------')
                        with nvtx.annotate('loss'):
                            #loss = loss_fcn(batch_pred, batch_labels)
                            loss = loss_fcn(batch_pred, batch.y)
                        with nvtx.annotate('zero_grad'):
                            optimizer.zero_grad()
                        with nvtx.annotate('backward', color='purple'):
                            loss.backward()
                        with nvtx.annotate('step'):
                            optimizer.step()

                with nvtx.annotate('log'):
                    iter_tput.append(batch_pred.size(0) / (time.time() - tic_step))
                    if step % args.log_every == 0:
                        #acc = compute_acc(batch_pred, batch_labels)
                        acc = compute_acc(batch_pred, batch.y)
                        gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
                        print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
                            epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:]), gpu_mem_alloc))
                    tic_step = time.time()

            toc = time.time()
            print('Epoch Time(s): {:.4f}'.format(toc - tic))
            if epoch >= 5:
                avg += toc - tic
            if epoch % args.eval_every == 0 and epoch != 0:
                with nvtx.annotate('validation'):
                    eval_acc = evaluate(model, g, x, y, val_nid, device)
                    print('Eval Acc {:.4f}'.format(eval_acc))
                with nvtx.annotate('test'):
                    test_acc = evaluate(model, g, x, y, test_nid, device)
                    print('Test Acc: {:.4f}'.format(test_acc))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--dataset', type=str, default='ogbn-products')
    argparser.add_argument('--num-epochs', type=int, default=3)
    argparser.add_argument('--num-hidden', type=int, default=16)
    argparser.add_argument('--num-layers', type=int, default=3)
    argparser.add_argument('--fan-out', type=str, default='5,10,15')
    #argparser.add_argument('--fan-out', type=str, default='2,3,4')
    argparser.add_argument('--batch-size', type=int, default=1024)
    #argparser.add_argument('--batch-size', type=int, default=4)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-workers', type=int, default=1,
                           help="Number of sampling processes. Use 0 for no extra process. "
                                "Default set to 1 so sampling and subgraph construction take "
                                "place on CPU for easier comparison of dataloaders.")
    argparser.add_argument('--data-cpu', action='store_true',
                           help="By default the script puts all node features and labels "
                                "on GPU when using it to save time for data copy. This may "
                                "be undesired if they cannot fit in GPU memory at once. "
                                "This flag disables that.")
    args = argparser.parse_args()

    if args.gpu >= 0:
        device = th.device('cuda:%d' % args.gpu)
    else:
        device = th.device('cpu')

    if args.dataset == 'reddit':
        g, n_classes = load_reddit()
    elif args.dataset == 'ogbn-products':
        g, n_classes = load_ogb('ogbn-products')
    else:
        raise Exception('unknown dataset')

    # think this renaming is unnecessary
    #train_g = val_g = test_g = g
    #train_nfeat = val_nfeat = test_nfeat = g.ndata.pop('features')
    #train_labels = val_labels = test_labels = g.ndata.pop('labels')
    x = g.ndata.pop('features')
    y = g.ndata.pop('labels')

    if not args.data_cpu:
        #train_nfeat = train_nfeat.to(device)
        #train_labels = train_labels.to(device)
        x = x.to(device)
        y = y.to(device)

    # Pack data
    #data = n_classes, train_g, val_g, test_g, train_nfeat, train_labels, \
    #       val_nfeat, val_labels, test_nfeat, test_labels
    data = n_classes, g, x, y

    run(args, device, data)

