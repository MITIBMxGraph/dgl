#ifndef FAST_SAMPLER_SESSION_H_
#define FAST_SAMPLER_SESSION_H_


#include "fs_common.h"


class FastSamplerSession {
 public:
  using Range = std::pair<int32_t, int32_t>;
  // using Chunk = std::pair<uint8_t, std::array<Range, 5>>;

  FastSamplerSession(
      size_t num_threads,
      unsigned int max_items_in_queue,
      FastSamplerConfig config_
  );

  ~FastSamplerSession() {
    // Return the threads to the pool.
    global_threadpool.consume(std::move(threads));
  }

  std::optional<PreparedSample> try_get_batch();

  std::optional<PreparedSample> blocking_get_batch();

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
  moodycamel::ProducerToken iptok; // input producer token
  moodycamel::ConsumerToken octok; // output consumer token

  // benchmarking data
  std::chrono::microseconds total_blocked_dur{};
  size_t total_blocked_occasions = 0;
};


#endif // FAST_SAMPLER_SESSION_H_
