#include "fsr_slot.hpp"


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
