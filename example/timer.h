#ifndef EXAMPLES_TIMER_H_
#define EXAMPLES_TIMER_H_

#include <chrono>

namespace examples {

using Milliseconds = std::chrono::milliseconds;

#define measure_in_milli(output) \
  for (examples::ScopedTimerState<examples::Milliseconds> state(output); \
       !state.stop; \
       state.stop = true)

template <typename T>
class ScopedTimer {
  using Clock = std::chrono::high_resolution_clock;
  using Time = std::chrono::high_resolution_clock::time_point;

 public:
  ScopedTimer(float& elapsed_time)
      : elapsed_time_(elapsed_time),
        start_time_(Clock::now()){
  }

  virtual ~ScopedTimer() {
    elapsed_time_ = std::chrono::duration_cast<T>(Clock::now() - start_time_).count();
  }

 private:
  float& elapsed_time_;
  const Time start_time_;
};

template <typename T>
struct ScopedTimerState {
  ScopedTimerState(float& elapsed_time)
      : timer(elapsed_time), stop(false) {}

  ScopedTimer<T> timer;
  bool stop;
};

}  // namespace examples

#endif  // EXAMPLES_TIMER_H_
