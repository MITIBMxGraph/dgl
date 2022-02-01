from typing import Optional

import torch
import torch.nn.functional as F

from .samplers import PreparedBatch
from .transferers import DeviceIterator
from .concepts import TrainCore, TrainCallback
from .utils import Timer
from .utils import CUDAAggregateTimer

#TRAIN_BACKWARD_TIME_START = torch.cuda.Event(enable_timing=True)
#TRAIN_BACKWARD_TIME_END = torch.cuda.Event(enable_timing=True)

#TRAIN_TOTAL_TIME = None

TRAIN_TIMER_LIST = []

def barebones_train_core(model: torch.nn.Module, batch: PreparedBatch):
    global TRAIN_TIMER_LIST
    TRAIN_MODEL_TIME_START = torch.cuda.Event(enable_timing=True)
    TRAIN_MODEL_TIME_END = torch.cuda.Event(enable_timing=True)
    TRAIN_MODEL_TIME_START.record()
    out = model(batch.x, batch.adjs)
    loss = F.nll_loss(out, batch.y)
    loss.backward()
    TRAIN_MODEL_TIME_END.record()
    TRAIN_TIMER_LIST.append((TRAIN_MODEL_TIME_START, TRAIN_MODEL_TIME_END))
    #torch.cuda.synchronize()
    #if TRAIN_TOTAL_TIME == None:
    #    TRAIN_TOTAL_TIME = TRAIN_MODEL_TIME_START.elapsed_time(TRAIN_MODEL_TIME_END)
    #else:
    #    TRAIN_TOTAL_TIME = TRAIN_TOTAL_TIME + TRAIN_MODEL_TIME_START.elapsed_time(TRAIN_MODEL_TIME_END)


def report_TRAIN_TIME():
    global TRAIN_TIMER_LIST
    #global TRAIN_TOTAL_TIME
    torch.cuda.synchronize()
    total_time = TRAIN_TIMER_LIST[0][0].elapsed_time(TRAIN_TIMER_LIST[0][1])
    for x in TRAIN_TIMER_LIST[1:]:
        total_time += x[0].elapsed_time(x[1])
    print("TRAIN TIME INSIDE TRAIN_CORE ONLY IS :" + str(total_time))
    TRAIN_TIMER_LIST = []

def make_eval_and_loss(module, train_core):
    def eval_and_loss(*args, **_):
        train_core(module, PreparedBatch(*args))

    return eval_and_loss


def data_parallel_train(model: torch.nn.Module,
                        train_core: TrainCore,
                        devit: DeviceIterator,
                        optimizer: torch.optim.Optimizer,
                        cb: Optional[TrainCallback] = None) -> None:
    model.train()

    def update_timer(aggregate_time_dict, timer_res):
        if timer_res.name in aggregate_time_dict:
            aggregate_time_dict[timer_res.name] += timer_res.nanos
        else:
            aggregate_time_dict[timer_res.name] = timer_res.nanos

    aggregate_time_dict = dict()
    aggregate_time_lambda = lambda timer_res : update_timer(aggregate_time_dict, timer_res)

    training_timer_list = []

    dp_list = []

    ctimer_train = CUDAAggregateTimer("train")
    ctimer_load_batch = CUDAAggregateTimer("load_batch")
    ctimer_total = CUDAAggregateTimer("total")

    dp_list.append(ctimer_train)
    dp_list.append(ctimer_load_batch)
    dp_list.append(ctimer_total)

    ctimer_total.start()
    while True:
        epoch = 4221

        ctimer_train.start()
        with Timer((epoch, 'Data parallel training 1'), aggregate_time_lambda):
            optimizer.zero_grad()

            # Replicate the model (send weights) to devices
            # TODO: This might not be non-blocking. If so, this is a PyTorch issue!
            # NOTE: This creates "replica modules" whose gradients are automatically
            #       reduced during the computation of the backward pass.
            replicas = torch.nn.parallel.replicate(model, devit.devices)
        ctimer_train.end()

        ctimer_load_batch.start()
        with Timer((epoch, 'Data parallel training 1.5'), aggregate_time_lambda):
            inputs = next(devit, [])
        ctimer_load_batch.end()

        if len(inputs) == 0:
            break

        ctimer_train.start()
        with Timer((epoch, 'Data parallel training 2'), aggregate_time_lambda):
            replicas = replicas[:len(inputs)]
            devices = devit.devices[:len(inputs)]

        with Timer((epoch, 'Data parallel training 2.5'), aggregate_time_lambda):
            funcs = [make_eval_and_loss(replica, train_core)
                     for replica in replicas]

        with Timer((epoch, 'Data parallel training 3'), aggregate_time_lambda):
            # NOTE: devices can be inferred from inputs, but providing them is faster
            results = torch.nn.parallel.parallel_apply(
                funcs, inputs, devices=devices)

        with Timer((epoch, 'Data parallel training 4'), aggregate_time_lambda):
            optimizer.step()


        with Timer((epoch, 'Data parallel training 5'), aggregate_time_lambda):
            if cb is not None:
                cb(inputs, results)

        # skip replicating next iter, if we have no more data
        if len(inputs) < len(devit.devices):
            ctimer_train.end()
            break

        with Timer((epoch, 'Data parallel training 6'), aggregate_time_lambda):
            del inputs
            del results
            del funcs
        ctimer_train.end()
    ctimer_total.end()
    for x in dp_list:
        x.report()

    for x in aggregate_time_dict.keys():
        aggregate_time_dict[x] = aggregate_time_dict[x] / 1000000.0
    print(aggregate_time_dict)
    print (sum([aggregate_time_dict[x] for x in aggregate_time_dict.keys()]))

def serial_train(model: torch.nn.Module,
                 train_core: TrainCore,
                 devit: DeviceIterator,
                 optimizer: torch.optim.Optimizer,
                 cb: Optional[TrainCallback] = None) -> None:
    model.train()

    for inputs in devit:
        # only one input
        inp, = inputs

        optimizer.zero_grad()
        result = train_core(model, inp)
        optimizer.step()

        if cb is not None:
            cb([inp], [result])
