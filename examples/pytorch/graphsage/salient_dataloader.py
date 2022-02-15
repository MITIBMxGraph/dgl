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

from collections import namedtuple

# half precision support
from torch.cuda.amp import autocast, GradScaler


use_fp16 = True


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

    # change to half precision
    # do not want to bloat communication / transfers unnecessarily when precision is not warranted
    if use_fp16:
        x = x.half()

    x_dim = x.shape[1]
    train_nid = th.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
    val_nid = th.nonzero(g.ndata['val_mask'], as_tuple=True)[0]
    test_nid = th.nonzero(~(g.ndata['train_mask'] | g.ndata['val_mask']), as_tuple=True)[0]

    # if want to be consistent with DGL's reversed notation must reverse fanout
    dgl__fanout = [int(f) for f in args.fan_out.split(',')]
    salient__fanout = list(reversed(dgl__fanout))

    #convert graph to input to format that can be fed into Salient
    with nvtx.annotate('convert DGLGraph to CSR for Salient'):
        rowptr, col, edge_ids = g.adj_sparse('csr')
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
    #train_loader = FastSampler(2, 50, cfg)
    #train_loader = FastSampler(1, 50, cfg) # serial to debug

    # Define model and optimizer
    model = SAGE(x_dim, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout)
    #model = model.half().to(device)
    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    compute_stream = torch.cuda.Stream(device)

    # Comparison loop
    iter_tput = []
    for epoch in range(args.num_epochs):
        with nvtx.annotate('epoch'):
            tic = time.time()
            tic_step = time.time()

            train_loader_iter = iter(train_loader)
            salient__dataloader = DevicePrefetcher([device], train_loader_iter)
            step = -1

            while True:

                #with torch.cuda.stream(compute_stream):
                with nvtx.annotate('iteration'):
                    step += 1
                    #print(step)

                    inputs = next(salient__dataloader, [])
                    if len(inputs) == 0:
                        break

                    # usually just one batch?
                    for batch in inputs:

                        # need this wait() call with dgl's AsyncTransferer
                        #batch = batch.wait()
                        #batch = Batch(x=prepared_batch.x.wait(), y=prepared_batch.y.wait(), blocks=prepared_batch.blocks.wait())

                        #print(f'batch.x.device():                   {batch.x.device}')
                        #print(f'batch.y.device():                   {batch.y.device}')
                        #for block in batch.blocks:
                            #print(f'block device:                        {block.device}')
                            #print(f"block._node_frames[0]['_ID'].device: {block._node_frames[0]['_ID'].device}")
                            #print(f"block._node_frames[1]['_ID'].device: {block._node_frames[1]['_ID'].device}")
                            #print(f"block._edge_frames[0]['_ID'].device: {block._edge_frames[0]['_ID'].device}")
                            #print(f"block._edge_frames[1]['_ID'].device: {block._edge_frames[1]['_ID'].device}")
                            #print(f"block._graph.ctx.device_type:        {block._graph.ctx.device_type}")

                        # Compute loss and prediction
                        with autocast(enabled=use_fp16):
                            with nvtx.annotate('model'):
                                #with th.autograd.profiler.emit_nvtx():
                                    #batch_pred = model(blocks, batch_inputs)
                                batch_pred = model(batch.blocks, batch.x)
                                with nvtx.annotate('unpinning'):
                                    pass
                                    # this doesn't make sense because pointing to gpu objects
                                    #for block in batch.blocks: block.unpin_memory_()
                            with nvtx.annotate('loss'):
                                #loss = loss_fcn(batch_pred, batch_labels)
                                loss = loss_fcn(batch_pred, batch.y)
                        with nvtx.annotate('zero_grad'):
                            optimizer.zero_grad()
                        with nvtx.annotate('backward', color='purple'):
                            loss.backward()
                        with nvtx.annotate('step'):
                            optimizer.step()

                        #print('---------- Salient Summary ----------')
                        #print(f'blocks: {batch.blocks}')
                        #print(f'sz batch.x, sz batch.y: {len(batch.x)}, {len(batch.y)}')
                        #print(f'sz batch_pred: {batch_pred.size()}')
                        #print('---------- --------------- ----------')

                        """
                        if use_fp16:
                            # Backprop w/ gradient scaling
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            optimizer.step()
                        """
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
    argparser.add_argument('--num-epochs', type=int, default=2)
    argparser.add_argument('--num-hidden', type=int, default=16)
    argparser.add_argument('--num-layers', type=int, default=3)
    argparser.add_argument('--fan-out', type=str, default='5,10,15')
    #argparser.add_argument('--fan-out', type=str, default='1,2,3')
    argparser.add_argument('--batch-size', type=int, default=1024)
    #argparser.add_argument('--batch-size', type=int, default=2)
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

