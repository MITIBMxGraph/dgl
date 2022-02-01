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

#### Entry point
def compare(args, device, data):
    # Unpack data
    n_classes, train_g, val_g, test_g, train_nfeat, train_labels, \
    val_nfeat, val_labels, test_nfeat, test_labels = data
    in_feats = train_nfeat.shape[1]
    train_nid = th.nonzero(train_g.ndata['train_mask'], as_tuple=True)[0]
    val_nid = th.nonzero(val_g.ndata['val_mask'], as_tuple=True)[0]
    test_nid = th.nonzero(~(test_g.ndata['train_mask'] | test_g.ndata['val_mask']), as_tuple=True)[0]

    fanout = [int(f) for f in args.fan_out.split(',')]

    # Create DGL DataLoader (similar to pytorch) for constructing blocks
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
    # create Salient dataloader
    cfg = FastSamplerConfig(
                # ignore features and labels temporarily
                # x=dataset.x, y=dataset.y.unsqueeze(-1),

                # old salient graph input (would have been created by driver.dataset.FastDataset)
                #rowptr=dataset.rowptr, col=dataset.col,
                #idx=dataset.split_idx['train'],

                # dgl compatible graph input
                graph=train_g,
                batch_size=args.batch_size, sizes=fanout,
                skip_nonfull_batch=False, pin_memory=True
            )

    train_loader = FastSampler(40,
                         50, cfg)
    train_loader_iter = iter(train_loader)
    salient__dataloader = train_loader_iter
    """


    # Comparison loop
    for epoch in range(args.num_epochs):

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        tic_step = time.time()
        dgl__dataloader_iter = iter(dgl__dataloader)
        step = -1
        while True:

            step += 1
            print(step)

            # load DGL prepared batch
            try:
                dgl__input_nodes, dgl__seeds, dgl__blocks = next(dgl__dataloader_iter)
            except StopIteration:
                # make sure DGL Dataloader and Salient complete simultaneously
                """
                inputs = next(salient__dataloader, [])
                assert len(inputs) == 0
                """
                break

            """
            # load Salient prepared batch
            inputs = next(salient__dataloader, [])
            assert len(inputs) != 0:
            """

            # add some more assertions on types returned by different dataloaders
            #print(step)
            #print(f'type dgl rv: {type(blocks)}')
            #print(f'type sal rv: {type(inputs)}')
            #print('blocks')
            print(dgl__blocks)
            #print('inputs')
            #print(inputs)
            #print()

            #print(dgl__blocks[2].__dict__)
           # print('coo')
            # adj_sparse gives wrong info
            #print(dgl__blocks[2].adj_sparse('coo'))
            print('returning')
            return

            """
            # Load the input features as well as output labels
            batch_inputs, batch_labels = load_subtensor(train_nfeat, train_labels,
                                                        seeds, input_nodes, device)
            blocks = [block.int().to(device) for block in blocks]
            """


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--dataset', type=str, default='ogbn-products')
    argparser.add_argument('--num-epochs', type=int, default=1)
    argparser.add_argument('--fan-out', type=str, default='4,3,2',
                           help="Number of neighbors to sample at each hop. "
                                "Kept small for easy visual comparison. ")
    argparser.add_argument('--batch-size', type=int, default=4,
                           help="Size of minibatch, kept tiny for easy visual "
                                "comparison.")
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
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

    train_g = val_g = test_g = g
    train_nfeat = val_nfeat = test_nfeat = g.ndata.pop('features')
    train_labels = val_labels = test_labels = g.ndata.pop('labels')

    if not args.data_cpu:
        train_nfeat = train_nfeat.to(device)
        train_labels = train_labels.to(device)

    # Pack data
    data = n_classes, train_g, val_g, test_g, train_nfeat, train_labels, \
           val_nfeat, val_labels, test_nfeat, test_labels

    compare(args, device, data)

