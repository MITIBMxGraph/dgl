from dgl._ffi.function import _init_api
from dgl._ffi.object import register_object, ObjectBase
from dgl.heterograph import DGLBlock

from abc import abstractmethod
import datetime
import itertools
from dataclasses import dataclass, fields
from collections.abc import Iterable, Iterator, Sized
from typing import List, Optional, NamedTuple

import torch
from torch_sparse import SparseTensor

# profling
import nvtx



_init_api("dgl.salient")


"""
Return types.
"""


@register_object('graph.HeteroGraphArray')
class HeteroGraphIndexArray(ObjectBase):
    """ An array of dgl::HeteroGraphRef objects. """

    def __getitem__(self, idx):
        return _CAPI_DGLHeteroArrayGetGraphAtIdx(self, idx)

    def __len__(self):
        return _CAPI_DGLHeteroArrayGetLen(self)


@register_object('salient.OptionalPreparedSample')
class OptionalPreparedSample(ObjectBase):
    """ An optional type containing:
        - batch features
        - batch labels (optional)
        - batch mfgs
        - range of indices trained on in batch
    """

    def __bool__(self):
        return True if _CAPI_OptionalPreparedSampleHasValue(self) else False

    @property
    def x(self):
       return _CAPI_OptionalPreparedSampleGetX(self) if _CAPI_OptionalPreparedSampleHasValue(self) else None

    @property
    def y(self):
       return _CAPI_OptionalPreparedSampleGetY(self) if (_CAPI_OptionalPreparedSampleHasValue(self) and _CAPI_OptionalPreparedSampleHasY(self)) else None

    @property
    def blocks(self):
        if _CAPI_OptionalPreparedSampleHasValue(self):
            mfgs = _CAPI_OptionalPreparedSampleGetMfgs(self)
            return [DGLBlock(mfgs[i]) for i in range(len(mfgs))]
        return None

    @property
    def range(self):
       return (_CAPI_OptionalPreparedSampleGetRangeStart(self), _CAPI_OptionalPreparedSampleGetRangeEnd(self)) if _CAPI_OptionalPreparedSampleHasValue(self) else None


"""
User-facing classes with C++ backend.
"""

@register_object('salient.FastSamplerConfig')
class FastSamplerConfig(ObjectBase):
    """ Configure the fast sampler. """

    #def __new__(cls, batch_size, x, y, rowptr, col, idx, sizes, skip_nonfull_batch, pin_memory):
    def create(batch_size, x, y, rowptr, col, idx, sizes, skip_nonfull_batch, pin_memory):
        return _CAPI_FSConfigCreate(batch_size, x, y, rowptr, col, idx, sizes, skip_nonfull_batch, pin_memory)

    @property
    def batch_size(self):
        return _CAPI_FSConfigGetBatchSize(self)

    @property
    def skip_nonfull_batch(self):
        return True if _CAPI_FSConfigSkipNonfullBatch(self) else False

    @property
    def num_train_idxs(self):
        return _CAPI_FSConfigGetNumTrainIdxs(self)

    def get_num_batches(self) -> int:
        num_batches, r = divmod(self.num_train_idxs, self.batch_size)
        if not self.skip_nonfull_batch and r > 0:
            num_batches += 1
        return num_batches


@register_object('salient.OptionalNDArray')
class OptionalNDArray(ObjectBase):

    #def __new__(cls, ndarray):
    def create(ndarray):
        return _CAPI_OptionalNDArrayCreate(ndarray)

    def __bool__(self):
        return True if _CAPI_OptionalNDArrayHasValue(self) else False

    @property
    def ndarray(self):
        return _CAPI_OptionalNDArrayGetNDArray(self)


@register_object('salient.FastSamplerSession')
class FastSamplerSession(ObjectBase):
    """ Main Fast Sampler class. """

    def create(num_threads, max_items_in_queue, cfg):
        return _CAPI_FSSessionCreate(num_threads, max_items_in_queue, cfg)

    @property
    def num_total_batches(self):

    def blocking_get_batch(self):
        return _CAPI_FSSessionBlockingGetBatch(self)


""" Class for working with returned batch. """


class PreparedBatch(NamedTuple):
    x: dgl.ndarray.NDArray
    y: Optional[dgl.ndarray.NDArray]
    blocks: List[DGLBlock]
    idx_range: slice

    @classmethod
    def from_fast_sampler(cls, batch):
        (start, stop) = batch.range
        return cls(
            x=batch.x,
            y=batch.y.squeeze() if batch.y is not None else None,
            blocks=batch.blocks,
            idx_range=slice(start, stop)
        )

    #def record_stream(self, stream):
    #    if self.x is not None:
    #        self.x.record_stream(stream)
    #    if self.y is not None:
    #        self.y.record_stream(stream)
    #    # record stream for DGLBlock
    #    """
    #    for block in self.blocks:
    #        block.int().record_stream(stream)
    #    """

    # TODO: reinstate after cleaned in DeviceTransferer
    #def to(self, device, non_blocking=False):
    #    # transfer blocks first
    #    tmp_blocks=[block.int().to(device=device, non_blocking=False) for block in self.blocks] if self.blocks is not None else None
    #    #torch.cuda.synchronize()
    #    return PreparedBatch(
    #        x=self.x.to(
    #            device=device, non_blocking=non_blocking) if self.x is not None else None,
    #        y=self.y.to(
    #            device=device, non_blocking=non_blocking) if self.y is not None else None,
    #        #blocks=[block.int().to(device=device, non_blocking=non_blocking)
    #        #      for block in self.blocks],
    #       # blocks=None,
    #        blocks=tmp_blocks,
    #        idx_range=self.idx_range
    #    )

    @property
    def num_total_nodes(self):
        return self.x.size(0)

    @property
    def batch_size(self):
        return self.idx_range.stop - self.idx_range.start


"""
Fast sampler python classes.
"""

class FastSamplerIter(Iterator[PreparedBatch]):
    session: FastSamplerSession

    def __init__(self, num_threads: int, max_items_in_queue: int, cfg: FastSamplerConfig):
        self.session = FastSanplerSession.create(
            num_threads, max_items_in_queue, cfg)
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
    def idx(self):
        ...


@dataclass
class FastSampler(ABCNeighborSampler):
    num_threads: int
    max_items_in_queue: int
    cfg: FastSamplerConfig

    @property
    def idx(self):
        return self.cfg.idx

    def __iter__(self):
        return FastSamplerIter(self.num_threads, self.max_items_in_queue, self.cfg)

    def __len__(self):
        return self.cfg.get_num_batches()

"""


"""
Test functions.
"""


def get_cpp_test_mfg():
    return DGLBlock(_CAPI_TestMFG())

def get_cpp_test_mfg_vector():
    print(_CAPI_TestMFGVector())
    return [DGLBlock(g) for g in _CAPI_TestMFGVector()]

def test_none_rv(x):
    return _CAPI_TestNone(x)
