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
          auto const tdo = nigiri::routing::td_offset{.valid_from_ = std::min(r.time_at_stop_, r.time_at_start_), .duration_ = std::chrono::abs(r.time_at_stop_ - r.time_at_start_), .transport_mode_id_ = mode};
          auto const i = std::lower_bound(begin(td_offsets.at(r.stop_)), end(td_offsets.at(r.stop_)), tdo, [](auto const& a, auto const& b) {
            return a.valid_from_ < b.valid_from_;
          });

          if (i == end(td_offsets.at(r.stop_)) || ) {
            td_offsets.at(r.stop_).insert(i,tdo);
          }
        }
      });
  return td_offsets;
}
} // namespace motis::odm