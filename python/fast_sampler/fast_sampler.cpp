#include <pybind11/chrono.h>
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
#include <optional>
#include <system_error>
#include <thread>
#include <string>

#include <semaphore.h> 
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

// just inluced the rest ha
#include <dgl/aten/coo.h>
#include <dgl/aten/csr.h>
#include <dgl/aten/macro.h>
#include <dgl/aten/spmat.h>
#include <dgl/aten/types.h>

#include <vector>
#include <tuple>
#include <utility>
//#include "../../array/cpu/array_utils.h"
// TODO: CLEANUP
#include "/home/gridsan/pmurzynowski/dgl/src/array/cpu/array_utils.h"
#include "/home/gridsan/pmurzynowski/dgl/src/graph/heterograph.h"
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

#include "concurrentqueue.h"
#include "sample_cpu.hpp"
#include "utils.hpp"

//using Adjs = std::vector<std::tuple<
//    torch::Tensor,
//    torch::Tensor,
//    torch::Tensor,
//    std::pair<int64_t, int64_t>>>;
//using ProtoSample = std::pair<torch::Tensor, Adjs>;
//using PreparedSample = std::tuple<torch::Tensor, std::optional<torch::Tensor>, Adjs, std::pair<int32_t, int32_t>>;

using Blocks = std::vector<dgl::HeteroGraphRef>;
// using Blocks = std::vector<dgl::runtime::DGLRetValue*>;
using ProtoSample = std::pair<torch::Tensor, Blocks>;
using PreparedSample = std::tuple<torch::Tensor, std::optional<torch::Tensor>, Blocks, std::pair<int32_t, int32_t>>; 


// Only sampling function that is guaranteed to be up to date
ProtoSample multilayer_sample(
    std::vector<int64_t> n_ids,
    std::vector<int64_t> const& sizes,
    torch::Tensor rowptr,
    torch::Tensor col,
    bool pin_memory = false) {
  auto n_id_map = get_initial_sample_adj_hash_map(n_ids);
  Blocks blocks;
  blocks.reserve(sizes.size());
  for (auto size : sizes) {
    auto const subset_size = n_ids.size();
    //torch::Tensor out_rowptr, out_col, out_e_id;
    dgl::IdArray out_rowptr, out_col, out_e_id;

    // sample_adj outputs a relation graph
    // for all intents and purposes this is the MFG for that layer
    std::tie(out_rowptr, out_col, n_ids, out_e_id) = sample_adj(
        rowptr, col, std::move(n_ids), n_id_map, size, false, pin_memory);
    // 'SRC/_N' and 'DST/_N'
    const int64_t nvtypes = 2;
    const int64_t num_dst = subset_size;
    const int64_t num_src = n_ids.size();
    std::cout << "num_dst: " << num_dst << std::endl;
    std::cout << "num_src: " << num_src << std::endl;
    std::cout << "out_rowptr: " << out_rowptr << std::endl;
    std::cout << "out_col: " << out_col << std::endl;
    std::cout << "out_e_id: " << out_e_id << std::endl;
    std::vector<dgl::SparseFormat> formats_vec = {dgl::ParseSparseFormat("csr")};
    const auto code = SparseFormatsToCode(formats_vec);
    auto hgptr = dgl::CreateFromCSR(nvtypes, num_src, num_dst,
                                    std::move(out_rowptr), std::move(out_col), std::move(out_e_id),
                                    code); 
    auto rel_graph = dgl::HeteroGraphRef(hgptr);

    // create metagraph
    constexpr DLContext ctx = DLContext{kDLCPU, 0};
    const uint8_t nbits = 64;
    // currently suming one type of relation graph, so the meta graph has only two nodes
    const int64_t num_nodes = 2;
    // src_ids contains node 0 of metagraph
    const int64_t num_src_ids = 1;
    dgl::IdArray src_ids = dgl::aten::NewIdArray(num_src_ids, ctx, nbits);
    src_ids.Ptr<int64_t>()[0] = 0;
    // dst_ids contains node 1 of metagraph
    const int64_t num_dst_ids = 1;
    dgl::IdArray dst_ids = dgl::aten::NewIdArray(num_dst_ids, ctx, nbits);
    dst_ids.Ptr<int64_t>()[0] = 1;
    // making readonly, so immutable
    auto metagraph = dgl::GraphRef(dgl::ImmutableGraph::CreateFromCOO(num_nodes, src_ids, dst_ids));

    // combine relation graph with metagraph
    // only have on relation graph and most simple metagraph
    std::vector<int64_t> num_nodes_per_type = {num_src, num_dst};
    std::vector<dgl::HeteroGraphPtr> rel_ptrs = {rel_graph.sptr()};
    auto out_hgptr = CreateHeteroGraph(metagraph.sptr(), rel_ptrs, num_nodes_per_type);
    auto out_graph_index = dgl::HeteroGraphRef(out_hgptr);

    blocks.emplace_back(out_graph_index);
  }

  std::reverse(blocks.begin(), blocks.end());
  return {vector_to_tensor(n_ids), std::move(blocks)};
}

ProtoSample multilayer_sample(
    torch::Tensor idx,
    std::vector<int64_t> const& sizes,
    torch::Tensor rowptr,
    torch::Tensor col,
    bool pin_memory = false) {
  const auto idx_data = idx.data_ptr<int64_t>();
  return multilayer_sample(
      {idx_data, idx_data + idx.numel()},
      sizes,
      std::move(rowptr),
      std::move(col),
      pin_memory);
}


template <typename scalar_t>
torch::Tensor serial_index_impl(
    torch::Tensor const in,
    torch::Tensor const idx,
    int64_t const n,
    bool const pin_memory = false) {
  const auto f = in.sizes().back();
  TORCH_CHECK(
      (in.strides().size() == 2 && in.strides().back() == 1) ||
          (in.sizes().back() == 1),
      "input must be 2D row-major tensor");

  torch::Tensor out =
      torch::empty({n, f}, in.options().pinned_memory(pin_memory));
  auto inptr = in.data_ptr<scalar_t>();
  auto outptr = out.data_ptr<scalar_t>();
  auto idxptr = idx.data_ptr<int64_t>();
  for (int64_t i = 0; i < std::min(idx.numel(), n); ++i) {
    const auto row = idxptr[i];
    std::copy_n(inptr + row * f, f, outptr + i * f);
  }

  return out;
}

template <typename scalar_t>
torch::Tensor serial_index_impl(
    torch::Tensor const in,
    torch::Tensor const idx,
    bool const pin_memory = false) {
  return serial_index_impl<scalar_t>(in, idx, idx.numel(), pin_memory);
}

torch::Tensor serial_index(
    torch::Tensor const in,
    torch::Tensor const idx,
    int64_t const n,
    bool const pin_memory = false) {
  return AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, in.scalar_type(), "serial_index", [&] {
    return serial_index_impl<scalar_t>(in, idx, n, pin_memory);
  });
}

torch::Tensor serial_index(
    torch::Tensor const in,
    torch::Tensor const idx,
    bool const pin_memory = false) {
  return AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, in.scalar_type(), "serial_index", [&] {
    return serial_index_impl<scalar_t>(in, idx, pin_memory);
  });
}

torch::Tensor to_row_major(torch::Tensor const in) {
  TORCH_CHECK(in.strides().size() == 2, "only support 2D tensors");
  auto const tr = in.sizes().front();
  auto const tc = in.sizes().back();

  if (in.strides().front() == tc && in.strides().back() == 1) {
    return in; // already in row major
  }

  TORCH_CHECK(
      in.strides().front() == 1 && tr == in.strides().back(),
      "input has unrecognizable stides");

  auto out = torch::empty_strided(in.sizes(), {tc, 1}, in.options());

  AT_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::Long, in.scalar_type(), "to_row_major", [&] {
        auto inptr = in.data_ptr<scalar_t>();
        auto outptr = out.data_ptr<scalar_t>();

        for (int64_t r = 0; r < tr; ++r) {
          for (int64_t c = 0; c < tc; ++c) {
            outptr[r * tc + c] = inptr[c * tr + r];
          }
        }
      });

  return out;
}

template <typename x_scalar_t, typename y_scalar_t>
std::vector<std::vector<PreparedSample>> full_sample_impl(
    torch::Tensor const x,
    torch::Tensor const y,
    torch::Tensor const rowptr,
    torch::Tensor const col,
    torch::Tensor const idx,
    int64_t const batch_size,
    std::vector<int64_t> sizes,
    bool const skip_nonfull_batch = false,
    bool const pin_memory = false) {
  CHECK_CPU(x);
  CHECK_CPU(y);
  CHECK_CPU(rowptr);
  CHECK_CPU(col);
  CHECK_CPU(idx);

  std::vector<std::vector<PreparedSample>> results(omp_get_max_threads());

  auto const n = idx.numel();
  const auto idx_data = idx.data_ptr<int64_t>();

#pragma omp parallel for schedule(dynamic)
  for (int64_t i = 0; i < n; i += batch_size) {
    auto const this_batch_size = std::min(n, i + batch_size) - i;
    if (skip_nonfull_batch && (this_batch_size < batch_size)) {
      continue;
    }

    const std::pair<int32_t, int32_t> pair = {i, i + this_batch_size};
    auto proto = multilayer_sample({idx_data + pair.first, idx_data + pair.second}, sizes, rowptr, col, pin_memory);
    auto const& n_id = proto.first;
    auto x_s = serial_index_impl<x_scalar_t>(x, n_id, pin_memory);
    auto y_s =
        serial_index_impl<y_scalar_t>(y, n_id, this_batch_size, pin_memory);
    results[omp_get_thread_num()].emplace_back(
        std::move(x_s), std::move(y_s), std::move(proto.second), std::move(pair));
  }

  return results;
}

std::vector<std::vector<PreparedSample>> full_sample(
    torch::Tensor const x,
    torch::Tensor const y,
    torch::Tensor const rowptr,
    torch::Tensor const col,
    torch::Tensor const idx,
    int64_t const batch_size,
    std::vector<int64_t> sizes,
    bool const skip_nonfull_batch = false,
    bool const pin_memory = false) {
  return AT_DISPATCH_ALL_TYPES(x.scalar_type(), "full_sample_x", [&] {
    using x_scalar_t = scalar_t;
    return AT_DISPATCH_ALL_TYPES(y.scalar_type(), "full_sample_y", [&] {
      using y_scalar_t = scalar_t;
      return full_sample_impl<x_scalar_t, y_scalar_t>(
          x,
          y,
          rowptr,
          col,
          idx,
          batch_size,
          sizes,
          skip_nonfull_batch,
          pin_memory);
    });
  });
}

template <class Slot>
class Thread {
 public:
  template <class Function>
  Thread(Function&& f, std::unique_ptr<Slot> slot_)
      : slot{std::move(slot_)},
        thread{std::forward<Function>(f), std::ref(*(slot.get()))} {}

  template <class Function>
  Thread(Function&& f)
      : Thread{std::forward<Function>(f), std::make_unique<Slot>()} {}

  Thread(Thread&&) = default;
  Thread& operator=(Thread&&) = default;

  ~Thread() {
    if (slot) {
      slot->decommission();
      while (!thread.joinable()) {
        std::this_thread::yield();
      }
      thread.join();
    }
  }

  // TODO: Hide this behind getter.
  std::unique_ptr<Slot> slot;
  std::thread thread;
};

template <class Thread>
class ThreadPool {
 public:
  ThreadPool(std::function<Thread()> thread_factory_, size_t max_size)
      : thread_factory{std::move(thread_factory_)}, max_size{max_size} {}
  ThreadPool(std::function<Thread()> thread_factory_)
      : ThreadPool{
            std::move(thread_factory_),
            std::numeric_limits<size_t>::max()} {}

  ~ThreadPool() {
    // Give notice ahead of time, to speed up shutdown
    for (auto& thread : pool) {
      thread.slot->decommission();
    }
  }

  // TODO: Make a container type that returns its threads to this pool on
  // destruction.
  std::vector<Thread> get(const size_t num) {
    std::vector<Thread> out;
    out.reserve(num);
    out.insert(
        out.end(),
        std::make_move_iterator(pool.end() - std::min(num, pool.size())),
        std::make_move_iterator(pool.end()));
    pool.erase(pool.end() - out.size(), pool.end());

    while (out.size() < num) {
      out.emplace_back(thread_factory());
    }
    return out;
  }

  void consume(std::vector<Thread> threads) {
    for (auto& thread : threads) {
      thread.slot->hibernate_begin();
    }

    // wait for successful hibernate
    for (auto& thread : threads) {
      thread.slot->hibernate_end();
    }

    pool.insert(
        pool.end(),
        std::make_move_iterator(threads.begin()),
        std::make_move_iterator(
            threads.begin() +
            std::min(max_size - pool.size(), threads.size())));
    // the remaining threads should get destructed
  }

 public:
  const std::function<Thread()> thread_factory;
  const size_t max_size;

 private:
  std::vector<Thread> pool;
};

class FastSamplerSession;
struct FastSamplerSlot {
  std::mutex mutex;
  std::condition_variable cv;

  FastSamplerSession* session = nullptr;
  moodycamel::ProducerToken optok{};

  // TODO: should these change to relaxed atomics?
  // we're willing to accept the thread trying to unsuccessfully dequeue
  // until the values propagate between cores
  // according to godbolt, both x86 gcc and clang are dumb
  // and make unnecessary test or and instructions when working with the atomic
  // The PowerPC instructions just look very funny in either case.
  // https://godbolt.org/z/sEYTcGP5o
  volatile bool hibernate_flag = true;
  volatile bool decommissioned = false;

  bool should_hibernate() const {
    return hibernate_flag;
  }

  bool should_decommission() const {
    return decommissioned;
  }

  void assign_session(FastSamplerSession& new_session);

  void hibernate_begin() {
    // stores are not reordered with stores
    // TODO: Specify memory order to make sure.
    hibernate_flag = true;
    sem_wake();
  }

  void hibernate_end() {
    const std::lock_guard<decltype(mutex)> lock(mutex);
    session = nullptr;
  }

  void decommission() {
    decommissioned = true;
    cv.notify_one();
    sem_wake();
  }

 private:
  void sem_wake();
};

using FastSamplerThread = Thread<FastSamplerSlot>;

FastSamplerThread thread_factory();
ThreadPool<FastSamplerThread> global_threadpool{
    thread_factory,
    std::thread::hardware_concurrency()};

struct FastSamplerConfig {
  size_t batch_size;
  torch::Tensor x;
  std::optional<torch::Tensor> y;
  torch::Tensor rowptr, col, idx;
  std::vector<int64_t> sizes;
  bool skip_nonfull_batch;
  bool pin_memory;
};

class FastSamplerSession {
 public:
  using Range = std::pair<int32_t, int32_t>;
  // using Chunk = std::pair<uint8_t, std::array<Range, 5>>;

  FastSamplerSession(
      size_t num_threads,
      unsigned int max_items_in_queue,
      FastSamplerConfig config_)
      : config{std::move(config_)},
        // TODO: Why doesn't this compile...
        // threads{global_threadpool.get(num_threads)},
        items_in_queue{std::min(max_items_in_queue, items_in_queue.max())},
        inputs{config.idx.numel() / config.batch_size + 1},
        outputs{config.idx.numel() / config.batch_size + 1},
        iptok{inputs},
        octok{outputs} {
    TORCH_CHECK(
        max_items_in_queue > 0,
        "max_items_in_queue (%llu) must be positive",
        max_items_in_queue);
    TORCH_CHECK(
        max_items_in_queue <= items_in_queue.max(),
        "max_items_in_queue (%llu) must be <= %ll",
        max_items_in_queue,
        items_in_queue.max());
    threads = global_threadpool.get(num_threads);
    for (FastSamplerThread& thread : threads) {
      thread.slot->assign_session(*this);
    }

    size_t const n = config.idx.numel();

    for (size_t i = 0; i < n; i += config.batch_size) {
      auto const this_batch_size = std::min(n, i + config.batch_size) - i;
      if (config.skip_nonfull_batch && (this_batch_size < config.batch_size)) {
        continue;
      }

      // TODO: Maybe convert to enqueue_batch
      num_total_batches++;
      inputs.enqueue(iptok, Range(i, i + this_batch_size));
    }
  }

  ~FastSamplerSession() {
    // Return the threads to the pool.
    global_threadpool.consume(std::move(threads));
  }

  std::optional<PreparedSample> try_get_batch() {
    if (num_consumed_batches == num_total_batches) {
      return {};
    }

    PreparedSample batch;
    if (!outputs.try_dequeue(octok, batch)) {
      return {};
    }
    num_consumed_batches++;
    items_in_queue.release();
    return batch;
  }


  std::optional<PreparedSample> blocking_get_batch() {
    if (num_consumed_batches == num_total_batches) {
      return {};
    }

    auto batch = try_get_batch();
    if (batch) {
      return batch;
    }

    auto start = std::chrono::high_resolution_clock::now();
    while (true) {
      auto batch = try_get_batch();
      if (batch) {
        auto end = std::chrono::high_resolution_clock::now();
        total_blocked_dur +=
            std::chrono::duration_cast<decltype(total_blocked_dur)>(
                end - start);
        total_blocked_occasions++;
        return batch;
      }
    }
  }

  size_t get_num_consumed_batches() const {
    return num_consumed_batches;
  }

  size_t get_approx_num_complete_batches() const {
    return num_consumed_batches + num_inserted_batches;
  }

  size_t get_num_total_batches() const {
    return num_total_batches;
  }

  const FastSamplerConfig config;

  std::atomic<size_t> num_inserted_batches{0};

 private:
  std::vector<FastSamplerThread> threads;
  size_t num_consumed_batches = 0;
  size_t num_total_batches = 0;

 public:
  // std::counting_semaphore<> items_in_queue;
  MySemaphore items_in_queue;

 public:
  moodycamel::ConcurrentQueue<Range> inputs;
  moodycamel::ConcurrentQueue<PreparedSample> outputs;
  moodycamel::ConcurrentQueue<PreparedSample> mfg_outputs;
  moodycamel::ProducerToken iptok; // input producer token
  moodycamel::ConsumerToken octok; // output consumer token

  // benchmarking data
  std::chrono::microseconds total_blocked_dur{};
  size_t total_blocked_occasions = 0;
};


/*
DGL_REGISTER_GLOBAL("salient._CAPI_dgl_blocking_get_mfg")
.set_body([] (dgl::runtime::DGLArgs args, dgl::runtime::DGLRetValue* rv) {
    // Only valid if using CPython, quite sketchy
    int x = 98;
    void *ses_ptr = args[0];
    FastSamplerSession *ses = static_cast<FastSamplerSession*>(ses_ptr);
    size_t num_consumed_batches = ses->get_num_consumed_batches();
    size_t num_total_batches = ses->get_num_total_batches();
    if (num_consumed_batches == num_total_batches) {
      *rv = NULL;
    } else {
      auto batch = ses->try_get_batch();
      if (batch) {
        *rv = std::get<2>(batch.value())[0];
      } else {
        auto start = std::chrono::high_resolution_clock::now();
        while (true) {
          auto batch = ses->try_get_batch();
          if (batch) {
            auto end = std::chrono::high_resolution_clock::now();
            ses->total_blocked_dur +=
                std::chrono::duration_cast<decltype(ses->total_blocked_dur)>(
                    end - start);
            ses->total_blocked_occasions++;
            *rv = std::get<2>(batch.value())[0];
          }
        }
      }
    }
  });
*/

void FastSamplerSlot::assign_session(FastSamplerSession& new_session) {
  std::unique_lock<decltype(mutex)> lock(mutex);
  session = &new_session;
  optok = moodycamel::ProducerToken{new_session.outputs};
  hibernate_flag = false;
  lock.unlock();
  cv.notify_one();
}

void FastSamplerSlot::sem_wake() {
  if (session == nullptr) {
    return;
  }
  session->items_in_queue.release();
}

void fast_sampler_thread(FastSamplerSlot& slot) {

  // annotatex with nvtx
  nvtxRangePushA("fast_sampler_thread");

  std::unique_lock<decltype(slot.mutex)> lock(slot.mutex);
  while (true) {
    if (slot.should_hibernate()) {
      slot.cv.wait(lock, [&slot] {
        return slot.should_decommission() || !slot.should_hibernate();
      });
    }

    if (slot.should_decommission()) {
      return;
    }

    std::pair<int32_t, int32_t> pair;
    if (!slot.session->inputs.try_dequeue_from_producer(
            slot.session->iptok, pair)) {
      // std::this_thread::yield();
      continue;
    }

    slot.session->items_in_queue.acquire();

    // check if we were woken just to decommission or hibernate
    if (slot.should_hibernate() || slot.should_decommission()) {
      continue;
    }

    auto const this_batch_size = pair.second - pair.first;

    auto const& config = slot.session->config;
    const auto idx_data = config.idx.data_ptr<int64_t>();
    nvtxRangePushA("multilayer_sample");
    auto proto = multilayer_sample(
        {idx_data + pair.first, idx_data + pair.second},
        config.sizes,
        config.rowptr,
        config.col,
        config.pin_memory);
    nvtxRangePop();
    auto const& n_id = proto.first;
    nvtxRangePushA("slicing");
    // printf("Slicing x\n");
    // std::cout << config.x.sizes() << std::endl;
    // std::cout << n_id.sizes() << std::endl;
    // std::cout << torch::max(n_id) << std::endl;
    // std::cout << torch::min(n_id) << std::endl;
    auto x_s = serial_index(config.x, n_id, config.pin_memory);
    std::optional<torch::Tensor> y_s;
    if (config.y.has_value()) {
      // printf("Slicing y\n");
      y_s = serial_index(*config.y, n_id, this_batch_size, config.pin_memory);
    }
    nvtxRangePop();

    // TODO: Implement limit on the size of the output queue,
    //       to avoid high memory consumption when outpacing the training code.
    slot.session->outputs.enqueue(
        slot.optok, {std::move(x_s), std::move(y_s), std::move(proto.second), std::move(pair)});
    ++slot.session->num_inserted_batches;
  }

  nvtxRangePop();
}

FastSamplerThread thread_factory() {
  return FastSamplerThread{fast_sampler_thread};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<FastSamplerConfig>(m, "Config")
      .def(py::init<>())
      .def_readwrite("x", &FastSamplerConfig::x)
      .def_readwrite("y", &FastSamplerConfig::y)
      .def_readwrite("rowptr", &FastSamplerConfig::rowptr)
      .def_readwrite("col", &FastSamplerConfig::col)
      .def_readwrite("idx", &FastSamplerConfig::idx)
      .def_readwrite("batch_size", &FastSamplerConfig::batch_size)
      .def_readwrite("sizes", &FastSamplerConfig::sizes)
      .def_readwrite(
          "skip_nonfull_batch", &FastSamplerConfig::skip_nonfull_batch)
      .def_readwrite("pin_memory", &FastSamplerConfig::pin_memory);
  py::class_<FastSamplerSession>(m, "Session")
      .def(
          py::init<size_t, unsigned int, FastSamplerConfig>(),
          py::arg("num_threads"),
          py::arg("max_items_in_queue"),
          py::arg("config"))
      .def_readonly("config", &FastSamplerSession::config)
      .def("try_get_batch", &FastSamplerSession::try_get_batch)
      .def("blocking_get_batch", &FastSamplerSession::blocking_get_batch)
      .def_property_readonly(
          "num_consumed_batches", &FastSamplerSession::get_num_consumed_batches)
      .def_property_readonly(
          "num_total_batches", &FastSamplerSession::get_num_total_batches)
      .def_property_readonly(
          "approx_num_complete_batches",
          &FastSamplerSession::get_approx_num_complete_batches)
      .def_readonly("total_blocked_dur", &FastSamplerSession::total_blocked_dur)
      .def_readonly(
          "total_blocked_occasions",
          &FastSamplerSession::total_blocked_occasions);
  m.def(
      "sample_adj",
      py::overload_cast<
          torch::Tensor,
          torch::Tensor,
          torch::Tensor,
          int64_t,
          bool,
          bool>(&sample_adj),
      "Sample the one-hop neighborhood of the batch nodes",
      py::arg("rowptr"),
      py::arg("col"),
      py::arg("idx"),
      py::arg("num_neighbors"),
      py::arg("replace"),
      py::arg("pin_memory") = false);
  m.def(
      "multilayer_sample",
      py::overload_cast<
          torch::Tensor,
          std::vector<int64_t> const&,
          torch::Tensor,
          torch::Tensor,
          bool>(&multilayer_sample),
      "Sample the multi-hop neighborhood of the batch nodes",
      py::arg("idx"),
      py::arg("sizes"),
      py::arg("rowptr"),
      py::arg("col"),
      py::arg("pin_memory") = false);
  m.def(
      "full_sample",
      &full_sample,
      "Parallel sample of the index divided into batch_size chunks",
      py::arg("x"),
      py::arg("y"),
      py::arg("rowptr"),
      py::arg("col"),
      py::arg("idx"),
      py::arg("batch_size"),
      py::arg("sizes"),
      py::arg("skip_nonfull_batch") = false,
      py::arg("pin_memory") = false);
  m.def("to_row_major", &to_row_major, "Convert 2D tensor to row major");
  m.def(
      "serial_index",
      py::overload_cast<torch::Tensor, torch::Tensor, bool>(&serial_index),
      "Extract the rows of in (2D) as specified by idx",
      py::arg("in"),
      py::arg("idx"),
      py::arg("pin_memory") = false);
  m.def(
      "serial_index",
      py::overload_cast<torch::Tensor, torch::Tensor, int64_t, bool>(
          &serial_index),
      "Extract the rows of in (2D) as specified by idx, up to n rows",
      py::arg("in"),
      py::arg("idx"),
      py::arg("n"),
      py::arg("pin_memory") = false);
}
