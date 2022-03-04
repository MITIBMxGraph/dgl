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
User-facing classes.
"""

@register_object('salient.FastSamplerConfig')
class FastSamplerConfig(ObjectBase):
    """ Configure the fast sampler. """

    # an __init__ may not exactly work
    def create():
        return _CAPI_FSConfigCreate()

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


"""

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
