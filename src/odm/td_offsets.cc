#include "motis/odm/td_offsets.h"

#include <ranges>

namespace motis::odm {

std::pair<nigiri::routing::td_offsets_t, nigiri::routing::td_offsets_t>
get_td_offsets_split(std::vector<nigiri::routing::offset> const& offsets,
                     std::vector<service_times_t> const& times,
                     nigiri::transport_mode_id_t const mode) {
  auto const split =
      offsets.empty()
          ? 0
          : std::distance(begin(offsets),
                          std::upper_bound(begin(offsets), end(offsets),
                                           offsets[offsets.size() / 2],
                                           [](auto const& a, auto const& b) {
                                             return a.duration_ < b.duration_;
                                           }));

  auto const offsets_lo = offsets | std::views::take(split);
  auto const times_lo = times | std::views::take(split);
  auto const offsets_hi = offsets | std::views::drop(split);
  auto const times_hi = times | std::views::drop(split);

  auto const derive_td_offsets = [&](auto const& offsets_split,
                                     auto const& times_split) {
    auto td_offsets = nigiri::routing::td_offsets_t{};
    for (auto const [o, t] : std::views::zip(offsets_split, times_split)) {
      td_offsets.emplace(o.target_, std::vector<nigiri::routing::td_offset>{});
      for (auto const& i : t) {
        td_offsets[o.target_].emplace_back(i.from_, o.duration_, mode);
        td_offsets[o.target_].emplace_back(
            i.to_, nigiri::footpath::kMaxDuration, mode);
      }
    }
    return td_offsets;
  };

  return std::pair{derive_td_offsets(offsets_lo, times_lo),
                   derive_td_offsets(offsets_hi, times_hi)};
}

}  // namespace motis::odm