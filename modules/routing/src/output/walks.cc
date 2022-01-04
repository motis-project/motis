#include "motis/routing/output/walks.h"

#include <algorithm>

#include "utl/verify.h"

namespace motis::routing::output {

void update_walk_times(std::vector<intermediate::stop>& stops,
                       std::vector<intermediate::transport> const& transports) {
  if (transports.empty()) {
    return;
  }

  for (auto const& t : transports) {
    if (!t.is_walk() || t.from_ == 0) {
      continue;
    }
    auto& from = stops[t.from_];
    auto& to = stops[t.to_];
    auto const delay = from.a_time_ - from.a_sched_time_;
    from.d_reason_ = from.a_reason_;
    from.d_sched_time_ = static_cast<time>(from.d_time_ - delay);
    to.a_reason_ = from.d_reason_;
    to.a_sched_time_ = static_cast<time>(to.a_time_ - delay);
  }

  auto const first_train = std::find_if(
      begin(transports), end(transports),
      [](intermediate::transport const& t) { return !t.is_walk(); });
  if (first_train == begin(transports) || first_train == end(transports)) {
    return;
  }

  for (auto i = std::distance(begin(transports), first_train); i != 0; --i) {
    auto const& t = transports[i - 1];
    utl::verify(t.is_walk(), "not a walk");
    auto& from = stops[t.from_];
    auto& to = stops[t.to_];
    auto const delay = to.d_time_ - to.d_sched_time_;
    to.a_reason_ = to.d_reason_;
    to.a_sched_time_ = static_cast<time>(to.a_time_ - delay);
    from.d_reason_ = to.a_reason_;
    from.d_sched_time_ = static_cast<time>(from.d_time_ - delay);
  }
}

}  // namespace motis::routing::output
