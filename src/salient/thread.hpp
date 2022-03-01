#ifndef FAST_SAMPLER_THREAD_CLASS_H_
#define FAST_SAMPLER_THREAD_CLASS_H_


#include "fs_common.hpp"


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

#endif // FAST_SAMPLER_THREAD_H_
