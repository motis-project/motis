#include "nigiri/types.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/start_times.h"

namespace motis::odm {

nigiri::routing::td_offsets_t get_td_offsets(
    auto const& rides,
    nigiri::transport_mode_id_t const mode) {
  using namespace std::chrono_literals;

  auto td_offsets = nigiri::routing::td_offsets_t{};
  utl::equal_ranges_linear(
      rides, [](auto const& a, auto const& b) { return a.stop_ == b.stop_; },
      [&](auto&& from_it, auto&& to_it) {
        td_offsets.emplace(from_it->stop_,
                           std::vector<nigiri::routing::td_offset>{});
        for (auto const& r : nigiri::it_range{from_it, to_it}) {
          auto const dep = std::min(r.time_at_stop_, r.time_at_start_);
          auto const dur = std::chrono::abs(r.time_at_stop_ - r.time_at_start_);
          if (td_offsets.at(from_it->stop_).size() > 1) {
            auto last = rbegin(td_offsets.at(from_it->stop_));
            auto const second_last = std::next(last);
            if (dep ==
                std::clamp(dep, second_last->valid_from_, last->valid_from_)) {
              // increase validity interval of last offset
              last->valid_from_ = dep + dur;
              continue;
            }
          }
          // add new offset
          td_offsets.at(from_it->stop_)
              .push_back({.valid_from_ = dep,
                          .duration_ = dur,
                          .transport_mode_id_ = mode});
          td_offsets.at(from_it->stop_)
              .push_back({.valid_from_ = dep + 1min,
                          .duration_ = nigiri::footpath::kMaxDuration,
                          .transport_mode_id_ = mode});
        }
      });
  return td_offsets;
}
} // namespace motis::odm