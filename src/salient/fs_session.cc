#include "fs_session.hpp"

ThreadPool<FastSamplerThread> global_threadpool{
  thread_factory,
  std::thread::hardware_concurrency()};

FastSamplerSession::FastSamplerSession(
    size_t num_threads,
    unsigned int max_items_in_queue,
    FastSamplerConfigRef config_)
    : config{std::move(config_)},
      // TODO: Why doesn't this compile...
      // threads{global_threadpool.get(num_threads)},
      items_in_queue{std::min(max_items_in_queue, items_in_queue.max())},
      inputs{config->idx.NumElements() / config->batch_size + 1},
      outputs{config->idx.NumElements() / config->batch_size + 1},
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

  size_t const n = config->idx.NumElements();

  for (size_t i = 0; i < n; i += config->batch_size) {
    auto const this_batch_size = std::min(n, i + config->batch_size) - i;
    if (config->skip_nonfull_batch && (this_batch_size < config->batch_size)) {
      continue;
    }

    // TODO: Maybe convert to enqueue_batch
    num_total_batches++;
    inputs.enqueue(iptok, Range(i, i + this_batch_size));
  }
}

FastSamplerSession::~FastSamplerSession() noexcept {
  // Return the threads to the pool.
  global_threadpool.consume(std::move(threads));
}


optional<PreparedSample> FastSamplerSession::try_get_batch() {
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


optional<PreparedSample> FastSamplerSession::blocking_get_batch() {
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


/* methods */

DGL_REGISTER_GLOBAL("salient._CAPI_FSSessionCreate")
.set_body([] (dgl::runtime::DGLArgs args, dgl::runtime::DGLRetValue* rv) {
    size_t num_threads = static_cast<size_t>(args[0]);
    // should be unsigned.. but should be fine
    int max_items_in_queue = args[1];
    FastSamplerConfigRef cfg = args[2];
    auto fss = std::make_shared<FastSamplerSession>(num_threads, max_items_in_queue, cfg);
    *rv = FastSamplerSessionRef(fss);
});

DGL_REGISTER_GLOBAL("salient._CAPI_FSSessionBlockingGetBatch")
.set_body([] (dgl::runtime::DGLArgs args, dgl::runtime::DGLRetValue* rv) {
    FastSamplerSessionRef fssr = args[0];
    auto opsr = std::make_shared<OptionalPreparedSample>();
    opsr->value = std::move(fssr->blocking_get_batch());
    *rv = opsr;
});

DGL_REGISTER_GLOBAL("salient._CAPI_FSSessionGetNumTotalBatches")
.set_body([] (dgl::runtime::DGLArgs args, dgl::runtime::DGLRetValue* rv) {
    FastSamplerSessionRef fssr = args[0];
    *rv = static_cast<int64_t>(fssr->get_num_total_batches());
});
