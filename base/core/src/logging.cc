#include "motis/core/common/logging.h"

namespace motis::logging {

std::string time(time_t const t) {
  char buf[sizeof "2011-10-08t07:07:09z-0430"];
  struct tm result {};
  MOTIS_GMT(&t, &result);
  auto const written =
      strftime(static_cast<char*>(buf), sizeof buf, "%FT%TZ%z", &result);
  (void)written;
  return buf;
}

std::string time() {
  time_t now = 0;
  (void)std::time(&now);
  return time(now);
}

scoped_timer::scoped_timer(std::string name)
    : name_{std::move(name)}, start_{std::chrono::steady_clock::now()} {
  LOG(info) << "[" << name_ << "] starting";
}

scoped_timer::~scoped_timer() {
  using namespace std::chrono;
  auto stop = steady_clock::now();
  double const t = duration_cast<microseconds>(stop - start_).count() / 1000.0;
  LOG(info) << "[" << name_ << "] finished"
            << " (" << t << "ms)";
}

manual_timer::manual_timer(std::string name)
    : name_{std::move(name)}, start_{std::chrono::steady_clock::now()} {
  LOG(info) << "[" << name_ << "] starting";
}

void manual_timer::stop_and_print() {
  using namespace std::chrono;
  stop_ = steady_clock::now();
  LOG(info) << "[" << name_ << "] finished"
            << " (" << duration_ms() << "ms)";
}

double manual_timer::duration_ms() const {
  using namespace std::chrono;
  return duration_cast<microseconds>(stop_ - start_).count() / 1000.0;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::mutex log::log_mutex_;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
bool log::enabled_ = true;

}  // namespace motis::logging
