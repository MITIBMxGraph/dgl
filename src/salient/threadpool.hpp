#ifndef FAST_SAMPLER_THREADPOOL_H_
#define FAST_SAMPLER_THREADPOOL_H_


#include "fs_common.h"


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


#endif // FAST_SAMPLER_THREADPOOL_H_
