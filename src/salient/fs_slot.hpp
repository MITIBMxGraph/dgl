#ifndef FAST_SAMPLER_SLOT_H
#define FAST_SAMPLER_SLOT_H


#include "fs_common.hpp"


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


#endif // FAST_SAMPLER_SLOT
