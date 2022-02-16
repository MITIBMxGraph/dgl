from abc import abstractmethod
import datetime
import itertools
from dataclasses import dataclass, fields
from collections.abc import Iterable, Iterator, Sized
from typing import List, Optional, NamedTuple

import torch
from torch_sparse import SparseTensor

import fast_sampler

from .monkeypatch import Adj


# dgl compatibility
import dgl
from dgl.heterograph import DGLBlock

# profling
import nvtx


def Adj__from_fast_sampler(adj) -> Adj:
    rowptr, col, e_id, sparse_sizes = adj
    return Adj(
        SparseTensor(rowptr=rowptr, row=None, col=col, value=None,
                     sparse_sizes=sparse_sizes, is_sorted=True, trust_data=True),
        e_id,
        sparse_sizes[::-1]
    )

@nvtx.annotate('adj to Block', color='cyan')
def Block__from_fast_sampler(adj) -> DGLBlock:
    #print('Block__from_fast_sampler')
    rowptr, col, edge_ids, sparse_sizes = adj
    # BUG: cloning here because the torch tensors allocated in fast sampler might be on the stack ://
    rowptr, col, edge_ids = torch.clone(rowptr), torch.clone(col), torch.clone(edge_ids)
    #print(f'rowptr adr: {hex(id(rowptr))}')
    #print(f'rowptr type: {type(rowptr)}')
    #print(f'rowptr dtype: {rowptr.dtype}')
    #print(f'rowptr numel: {rowptr.numel()}')
    #print(f'rowptr is_pinned: {rowptr.is_pinned()}')
    #print(f'rowptr device: {rowptr.device}')
    #print(f'rowptr: {rowptr}')
    #print(f'col: {col}')
    #print(f'edge_ids: {edge_ids}')
    # for dgl no longer need to create Adj and SparseTensor
    with nvtx.annotate('adj to Block MFG conversion'):
        #with nvtx.annotate('adj extract csr'):
        #    rowptr, col, edge_ids = adj.adj_t.csr()
        edge_ids = torch.empty(0) if edge_ids is None else edge_ids
        with nvtx.annotate('create_block from csr'):
            block = dgl.create_block(('csr', (rowptr, col, edge_ids)),
                                     num_src_nodes=sparse_sizes[0],
                                     num_dst_nodes=sparse_sizes[1])
                                     #num_dst_nodes=sparse_sizes[1], device=torch.device('cpu'))
        with nvtx.annotate('_node_frames _ID torch.arange()'):
            # pass in a flag if should pin?
            block._node_frames[0]['_ID'] = torch.arange(block.number_of_src_nodes()).pin_memory()
            block._node_frames[1]['_ID'] = torch.arange(block.number_of_dst_nodes()).pin_memory()
            #block._node_frames[0]['_ID'] = torch.arange(block.number_of_src_nodes())
            #block._node_frames[1]['_ID'] = torch.arange(block.number_of_dst_nodes())
    # DEBUG, make sure all component tensors are in pinned memory:
    # 1. Check graph structure
    # 2. check features (are features empty here? Or is this an extra copy of features!?)
    """
    print('nframes')
    for nframe in block._node_frames:
        print(type(nframe))
        print(nframe.__dict__)
        print(nframe['_ID'])
        print(nframe.size())
        print(nframe.is_pinned())
    for nframe in block._node_frames:
        print(nframe.size())
        print(nframe.is_pinned())
    # 3. check misc info
    """
    with nvtx.annotate("pinning", color="green"):
        block.pin_memory_()
    return block


class ProtoBatch(NamedTuple):
    n_id: torch.Tensor
    adjs: List[Adj]
    idx_range: slice

    @classmethod
    def from_fast_sampler(cls, proto_sample, idx_range):
        n_id, adjs = proto_sample
        adjs = Adj__from_fast_sampler(adjs)
        return cls(n_id=n_id, adjs=adjs, idx_range=idx_range)

    @property
    def batch_size(self):
        return self.idx_range.stop - self.idx_range.start


class PreparedBatch(NamedTuple):
    x: torch.Tensor
    y: Optional[torch.Tensor]
    blocks: List[DGLBlock]
    idx_range: slice

    @classmethod
    def from_proto_batch(cls, x: torch.Tensor, y: Optional[torch.Tensor], proto_batch: ProtoBatch):
        raise NotImplementedError
        """
        return cls(
            x=x[proto_batch.n_id],
            y=y[proto_batch.n_id[:proto_batch.batch_size]
                ] if y is not None else None,
            adjs=proto_batch.adjs,
            idx_range=proto_batch.idx_range
        )
        """

    @classmethod
    def from_fast_sampler(cls, prepared_sample):
        x, y, blocks, (start, stop) = prepared_sample
        return cls(
            x=x,
            y=y.squeeze() if y is not None else None,
            blocks=[DGLBlock(graph_index) for graph_index in blocks],
            idx_range=slice(start, stop)
        )

    def record_stream(self, stream):
        if self.x is not None:
            self.x.record_stream(stream)
        if self.y is not None:
            self.y.record_stream(stream)
        # record stream for DGLBlock
        """
        for block in self.blocks:
            block.int().record_stream(stream)
        """

    def to(self, device, non_blocking=False):
        # transfer blocks first
        tmp_blocks=[block.int().to(device=device, non_blocking=False) for block in self.blocks] if self.blocks is not None else None
        #torch.cuda.synchronize()
        return PreparedBatch(
            x=self.x.to(
                device=device, non_blocking=non_blocking) if self.x is not None else None,
            y=self.y.to(
                device=device, non_blocking=non_blocking) if self.y is not None else None,
            #blocks=[block.int().to(device=device, non_blocking=non_blocking)
            #      for block in self.blocks],
           # blocks=None,
            blocks=tmp_blocks,
            idx_range=self.idx_range
        )

    @property
    def num_total_nodes(self):
        return self.x.size(0)

    @property
    def batch_size(self):
        return self.idx_range.stop - self.idx_range.start


@dataclass
class FastSamplerConfig:
    x: torch.Tensor
    y: torch.Tensor
    rowptr: torch.Tensor
    col: torch.Tensor
    idx: torch.Tensor
    # dgl-compatible graph input
    #graph: DGLGraph
    batch_size: int
    sizes: List[int]
    skip_nonfull_batch: bool
    pin_memory: bool

    def to_fast_sampler(self) -> fast_sampler.Config:
        c = fast_sampler.Config()
        for field in fields(self):
            setattr(c, field.name, getattr(self, field.name))

        return c

    def get_num_batches(self) -> int:
        num_batches, r = divmod(self.idx.numel(), self.batch_size)
        if not self.skip_nonfull_batch and r > 0:
            num_batches += 1
        return num_batches


class FastSamplerStats(NamedTuple):
    total_blocked_dur: datetime.timedelta
    total_blocked_occasions: int

    @classmethod
    def from_session(cls, session: fast_sampler.Session):
        return cls(total_blocked_dur=session.total_blocked_dur,
                   total_blocked_occasions=session.total_blocked_occasions)


class FastSamplerIter(Iterator[PreparedBatch]):
    session: fast_sampler.Session

    def __init__(self, num_threads: int, max_items_in_queue: int, cfg: FastSamplerConfig):
        ncfg = cfg.to_fast_sampler()
        self.session = fast_sampler.Session(
            num_threads, max_items_in_queue, ncfg)
        assert self.session.num_total_batches == cfg.get_num_batches()

    def __next__(self):
        sample = self.session.blocking_get_batch()
        if sample is None:
            raise StopIteration

        return PreparedBatch.from_fast_sampler(sample)

    def get_stats(self) -> FastSamplerStats:
        return FastSamplerStats.from_session(self.session)


class ABCNeighborSampler(Iterable[PreparedBatch], Sized):
    @property
    @abstractmethod
    def idx(self) -> torch.Tensor:
        ...

    @idx.setter
    @abstractmethod
    def idx(self, idx: torch.Tensor) -> None:
        ...


@dataclass
class FastSampler(ABCNeighborSampler):
    num_threads: int
    max_items_in_queue: int
    cfg: FastSamplerConfig

    @property
    def idx(self):
        return self.cfg.idx

    @idx.setter
    def idx(self, idx: torch.Tensor) -> None:
        self.cfg.idx = idx

    def __iter__(self):
        return FastSamplerIter(self.num_threads, self.max_items_in_queue, self.cfg)

    def __len__(self):
        return self.cfg.get_num_batches()


@dataclass
class FastPreSampler(ABCNeighborSampler):
    cfg: FastSamplerConfig

    @property
    def idx(self):
        return self.cfg.idx

    @idx.setter
    def idx(self, idx: torch.Tensor) -> None:
        self.cfg.idx = idx

    def __iter__(self) -> Iterator[PreparedBatch]:
        cfg = self.cfg
        p = fast_sampler.full_sample(cfg.x, cfg.y, cfg.rowptr, cfg.col, cfg.idx,
                                     cfg.batch_size, cfg.sizes, cfg.skip_nonfull_batch, cfg.pin_memory)
        return (PreparedBatch.from_fast_sampler(sample) for sample in itertools.chain(*p))

    def __len__(self):
        return self.cfg.get_num_batches()
