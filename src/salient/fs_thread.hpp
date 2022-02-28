#ifndef FAST_SAMPLER_THREAD_H_
#define FAST_SAMPLER_THREAD_H_


#include "fs_common.hpp"


using FastSamplerThread = Thread<FastSamplerSlot>;

FastSamplerThread thread_factory();
ThreadPool<FastSamplerThread> global_threadpool{
    thread_factory,
    std::thread::hardware_concurrency()};


#endif // FAST_SAMPLER_THREAD_H
