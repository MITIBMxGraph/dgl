/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef DGL_RPC_TENSORPIPE_QUEUE_H_
#define DGL_RPC_TENSORPIPE_QUEUE_H_

#include <condition_variable>
#include <deque>
#include <mutex>

// include for nvtx profiling
#ifndef NO_NVTX
#include "nvToolsExt.h"

// see bottom of this page for reference: https://htmlcolorcodes.com/
// const uint32_t colors[] = {0xffffffff, 0xC0C0C0ff, 0x808080ff, 0x000000ff, 0xff0000ff, 0x800000ff, 0xffff00ff, 0x808000ff, 0x00ff00ff, 0x008000ff, 0x00ffffff, 0x008080ff, 0x0000ffff, 0x000080ff, 0xff00ffff, 0x800080ff};
// these above may be wrong
const uint32_t colors[] = {0xffffffff, 0xff000000, 0x808080ff, 0x000000ff, 0xff0000ff, 0x800000ff, 0xffff00ff, 0x808000ff, 0x00ff00ff, 0x008000ff, 0x00ffffff, 0x008080ff, 0x0000ffff, 0x000080ff, 0xff000000, 0x800080ff};
const int num_colors = sizeof(colors)/sizeof(uint32_t);

#define PUSH_RANGE(name,cid) { \
    int color_id = cid; \
    color_id = color_id%num_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxRangePushEx(&eventAttrib); \
}
#define POP_RANGE nvtxRangePop();
#else
#define PUSH_RANGE(name,cid)
#define POP_RANGE
#endif

namespace dgl {
namespace rpc {

template <typename T>
class Queue {
 public:
  // Capacity isn't used actually
  explicit Queue(int capacity = 1) : capacity_(capacity) {}

  void push(T t) {
    std::unique_lock<std::mutex> lock(mutex_);
    // while (items_.size() >= capacity_) {
    //   cv_.wait(lock);
    // }
    items_.push_back(std::move(t));
    cv_.notify_all();
  }

  T pop() {
    std::unique_lock<std::mutex> lock(mutex_);

    PUSH_RANGE("wait, lock", 1);
    while (items_.size() == 0) {
      cv_.wait(lock);
    }
    POP_RANGE

    T t(std::move(items_.front()));
    items_.pop_front();
    cv_.notify_all();
    return t;
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  const int capacity_;
  std::deque<T> items_;
};
}  // namespace rpc
}  // namespace dgl

#endif  // DGL_RPC_TENSORPIPE_QUEUE_H_
