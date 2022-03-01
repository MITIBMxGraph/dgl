#ifndef FAST_SAMPLER_THREAD_H_
#define FAST_SAMPLER_THREAD_H_


#include "fs_common.hpp"
#include "thread.hpp"
#include "threadpool.hpp"
#include "fs_slot.hpp"


using FastSamplerThread = Thread<FastSamplerSlot>;

extern FastSamplerThread thread_factory();
extern ThreadPool<FastSamplerThread> global_threadpool;


#endif // FAST_SAMPLER_THREAD_H
