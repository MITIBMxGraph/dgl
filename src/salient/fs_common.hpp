#ifndef FAST_SAMPLER_COMMON_H_
#define FAST_SAMPLER_COMMON_H_

#include <semaphore.h>
#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <system_error>
#include <thread>
#include <string>
#include <cstdint>

// not with c++14
// #include <optional>
#include <boost/optional.hpp>
using boost::optional;

// includes for nvtx profiling
#include "nvToolsExt.h"

// likley too many includes here, were needed for development
#include <dgl/base_heterograph.h>
#include <dgl/transform.h>
#include <dgl/array.h>
#include <dgl/runtime/ndarray.h>
#include <dgl/packed_func_ext.h>
#include <dgl/immutable_graph.h>

#include <dgl/runtime/registry.h>
#include <dgl/runtime/packed_func.h>
#include <dgl/runtime/container.h>
#include <dgl/runtime/object.h>
#include <dgl/runtime/c_runtime_api.h>

#include <dgl/aten/array_ops.h>
#include <dgl/aten/coo.h>
#include <dgl/aten/csr.h>
#include <dgl/aten/macro.h>
#include <dgl/aten/spmat.h>
#include <dgl/aten/types.h>

#include <vector>
#include <tuple>
#include <utility>

// TODO: CLEANUP
#include "/home/gridsan/pmurzynowski/dgl/src/array/cpu/array_utils.h"
#include "/home/gridsan/pmurzynowski/dgl/src/graph/heterograph.h"

#include "concurrentqueue.h"
#include "utils.hpp"


using namespace dgl::runtime;
using namespace dgl::aten;


struct MySemaphore {
  MySemaphore(unsigned int value) {
    sem_init(&sem, 0, value);
  }

  ~MySemaphore() noexcept(false) {
    if (sem_destroy(&sem) == -1) {
      throw std::system_error(errno, std::generic_category());
    }
  }

  void acquire() {
    if (sem_wait(&sem) == -1) {
      throw std::system_error(errno, std::generic_category());
    }
  }

  void release(std::ptrdiff_t update = 1) {
    while (update > 0) {
      if (sem_post(&sem) == -1) {
        throw std::system_error(errno, std::generic_category());
      }
      update--;
    }
  }

  static constexpr auto max() {
    return std::numeric_limits<unsigned int>::max();
  }

  sem_t sem;
};


#endif // FAST_SAMPLER_COMMON_H_
