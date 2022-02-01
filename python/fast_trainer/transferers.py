from typing import List
from collections.abc import Iterator
from collections import namedtuple
import torch

from .samplers import ProtoBatch, PreparedBatch


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
        self.next = []
        self.preload()
        #self.DT_TOTAL_WAIT_TIME = None
        self.DT_WAIT_TIMER_LIST = []


    def preload(self):
        self.next = []
        for device, stream in zip(self.devices, self.streams):
            batch = next(self.it, None)
            if batch is None:
                break

            with torch.cuda.stream(stream):
                self.next.append(batch.to(device, non_blocking=True))


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
