import time
from contextlib import ContextDecorator
from typing import NamedTuple
import torch

class TimerResult(NamedTuple):
    name: str
    nanos: int

    def __str__(self):
        return f'{self.name} took {self.nanos / 1e9} sec'

class CUDAAggregateTimer:
    def __init__(self, name: str):
        self.name = name
        self.timer_list = []
        self._start = None
        self._end = None
    def start(self):
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)
        self._start.record()
    def end(self):
        self._end.record()
        self.timer_list.append((self._start,self._end))
    def report(self):
        torch.cuda.synchronize()
        total_time = self.timer_list[0][0].elapsed_time(self.timer_list[0][1])
        for x in self.timer_list[1:]:
           total_time += x[0].elapsed_time(x[1])
        print("CUDA Aggregate (" + self.name + "):" + str(total_time))


class Timer(ContextDecorator):
    def __init__(self, name: str, fn=print):
        self.name = name
        self._fn = fn

    def __enter__(self):
        self.start_ns = time.perf_counter_ns()
        self.stop_ns = None
        return self

    def stop(self):
        self.stop_ns = time.perf_counter_ns()

    def __exit__(self, *_):
        if self.stop_ns is None:
            self.stop()

        nanos = self.stop_ns - self.start_ns
        self._fn(TimerResult(self.name, nanos))
