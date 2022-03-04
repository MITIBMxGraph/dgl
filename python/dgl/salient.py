import dgl
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
        return _CAPI_FSSessionGetNumTotalBatches(self)

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

    #def get_stats(self) -> FastSamplerStats:
    #    return FastSamplerStats.from_session(self.session)


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


""" Classes for transferring to gpu. """


class DeviceIterator(Iterator[List[PreparedBatch]]):
    '''
    Abstract class that returns PreparedBatch on devices (GPUs)
    '''
    devices: List[torch.cuda.device]

    def __init__(self, devices):
        assert len(devices) > 0
        self.devices = devices


class DevicePrefetcher(DeviceIterator):
    def __init__(self, devices, it: Iterator[PreparedBatch]):
        super().__init__(devices)
        self.it = it
        self.streams = [torch.cuda.Stream(device) for device in devices]
        # clean up, device prefetcher should be generic and not have a specified return value
        # rv should be specified by the iterator
        self.rv = namedtuple('batch', ['x', 'y', 'blocks'])
        self.next = []
        self.preload()
        #self.DT_TOTAL_WAIT_TIME = None
        self.DT_WAIT_TIMER_LIST = []

    @nvtx.annotate('preload', color='yellow')
    def preload(self):
        self.next = []
        for device, stream in zip(self.devices, self.streams):
        #for device, transferer in zip(self.devices, self.transferers):
            batch = next(self.it, None)
            if batch is None:
                break
            # default
            #with torch.cuda.stream(stream):
            #    self.next.append(batch.to(device, non_blocking=True))

            # cannout use single line like this, cleanup with function later
            #self.next.append(transferer.async_copy(batch, torch.device(device)))
            """
            print(f'x is pinned: {batch.x.is_pinned()}')
            print(f'y is pinned: {batch.y.is_pinned()}')
            [print(type(block)) for block in batch.blocks]
            [print(type(block.int())) for block in batch.blocks]
            [print(block.int().__dict__) for block in batch.blocks]
            x_gpu = transferer.async_copy(batch.x, torch.device(device))
            y_gpu = transferer.async_copy(batch.y, torch.device(device))
            blocks_gpu = [transferer.async_copy(block.int(), torch.device(device)) for block in batch.blocks]
            self.next.append(self.rv(x=x_gpu, y=y_gpu, blocks=blocks_gpu))
            print('appended')
            """
            #gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
            #print(f'gpu mem before: {gpu_mem_alloc}')
            #use default stream for blocks, blocking (but blocks are typically much smaller than features)
            with nvtx.annotate('sending blocks to device', color='purple'):
                with torch.autograd.profiler.emit_nvtx():
                    #blocks_gpu = [block.int().to(device, non_blocking=False) for block in batch.blocks]
                    #blocks_gpu = [block.to(device, non_blocking=False) for block in batch.blocks]
                    blocks_gpu = [block.to(device, non_blocking=True) for block in batch.blocks]
            #with nvtx.annotate('just after async', color='red'):
            #    print('just after async transfer call!')
                #[print(block.device) for block in blocks_gpu]
            #torch.cuda.synchronize()
            with torch.cuda.stream(stream):
                with nvtx.annotate('sending x and y to device', color='orange'):
                    x_gpu = batch.x.to(device=device, non_blocking=True)
                    y_gpu = batch.y.to(device=device, non_blocking=True)
            #blocks_gpu = [block.int().to(device, non_blocking=False) for block in batch.blocks]
            with nvtx.annotate('returning PreparedBatch', color='black'):
                self.next.append(PreparedBatch(x=x_gpu, y=y_gpu, blocks=blocks_gpu, idx_range=batch.idx_range))
            #gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
            #print(f'gpu mem after: {gpu_mem_alloc}')


    def __next__(self):

        cur_streams = [torch.cuda.current_stream(
            device) for device in self.devices]


        DT_WAIT_TIMER_START = torch.cuda.Event(enable_timing=True)
        DT_WAIT_TIMER_END = torch.cuda.Event(enable_timing=True)
        DT_WAIT_TIMER_START.record()
        for cur_stream, stream in zip(cur_streams, self.streams):
            cur_stream.wait_stream(stream)
        DT_WAIT_TIMER_END.record()
        self.DT_WAIT_TIMER_LIST.append((DT_WAIT_TIMER_START, DT_WAIT_TIMER_END))
        #torch.cuda.synchronize()
        #if self.DT_TOTAL_WAIT_TIME == None:
        #    self.DT_TOTAL_WAIT_TIME = DT_WAIT_TIMER_START.elapsed_time(DT_WAIT_TIMER_END)
        #else:
        #    self.DT_TOTAL_WAIT_TIME = self.DT_TOTAL_WAIT_TIME + DT_WAIT_TIMER_START.elapsed_time(DT_WAIT_TIMER_END)
        ret = self.next
        if not ret:
            torch.cuda.synchronize()
            total_wait_time = self.DT_WAIT_TIMER_LIST[0][0].elapsed_time(self.DT_WAIT_TIMER_LIST[0][1])
            counter = 0
            for timer_pair in self.DT_WAIT_TIMER_LIST[1:]:
                batch_delay_time = timer_pair[0].elapsed_time(timer_pair[1])
                #print("BATCHDELAY("+str(counter)+"):" + str(batch_delay_time))
                total_wait_time += batch_delay_time
            print("TIME WAITING ON DATA TRANSFERS: " + str(total_wait_time))
            self.DT_WAIT_TIMER_LIST = []
            raise StopIteration

        # TODO: this might be a bit incorrect
        # in theory, we want to record this event after all the training computation on
        # the default stream
        for cur_stream, batch in zip(cur_streams, ret):
            batch.record_stream(cur_stream)

        self.preload()
        return ret



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
