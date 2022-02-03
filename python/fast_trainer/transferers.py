from typing import List
from collections.abc import Iterator
from collections import namedtuple
import torch

from .samplers import ProtoBatch, PreparedBatch

#profiling
import nvtx

import dgl


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
        self.transferers = [dgl.dataloading.AsyncTransferer(torch.device(device)) for device in devices]

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
                blocks_gpu = [block.int().to(device, non_blocking=False) for block in batch.blocks]
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


class DeviceTransferer(DeviceIterator):
    def __init__(self, devices, it: Iterator[PreparedBatch]):
        super().__init__(devices)

        self.it = it

    def __next__(self):
        ret = [batch.to(device, non_blocking=True)
               for device, batch in zip(self.devices, self.it)]

        if len(ret) == 0:
            raise StopIteration

        return ret


class DeviceSlicerTransferer(DeviceIterator):
    # NOTE: This class only exists to provide functionality
    #       that we used to have and no longer need (DATA_ON_MAIN).
    #       You likely do not need to use this.
    # NOTE: x and y can be GPU tensors too!
    def __init__(self, devices, x: torch.Tensor, y: torch.Tensor, it: Iterator[ProtoBatch]):
        super().__init__(devices)

        self.x = x
        self.y = y
        self.it = it

    def __next__(self):
        ret = [PreparedBatch.from_proto_batch(self.x, self.y, proto_batch).to(device, non_blocking=True)
               for device, proto_batch in zip(self.devices, self.it)]

        if len(ret) == 0:
            raise StopIteration

        return ret
