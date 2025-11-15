#pragma once

#include "nigiri/routing/query.h"
#include "nigiri/routing/start_times.h"
#include "nigiri/types.h"

#include "motis/odm/prima.h"

namespace motis::odm {

nigiri::routing::td_offsets_t get_td_offsets(
    auto const& rides, nigiri::transport_mode_id_t const mode) {
  using namespace std::chrono_literals;

  auto td_offsets = nigiri::routing::td_offsets_t{};
  utl::equal_ranges_linear(
      rides, [](auto const& a, auto const& b) { return a.stop_ == b.stop_; },
      [&](auto&& from_it, auto&& to_it) {
        td_offsets.emplace(from_it->stop_,
                           std::vector<nigiri::routing::td_offset>{});

        for (auto const& r : nigiri::it_range{from_it, to_it}) {
          auto const tdo = nigiri::routing::td_offset{
              .valid_from_ = std::min(r.time_at_stop_, r.time_at_start_),
              .duration_ = std::chrono::abs(r.time_at_stop_ - r.time_at_start_),
              .transport_mode_id_ = mode};
          auto i = std::lower_bound(begin(td_offsets.at(r.stop_)),
                                    end(td_offsets.at(r.stop_)), tdo,
                                    [](auto const& a, auto const& b) {
                                      return a.valid_from_ < b.valid_from_;
                                    });

          if (i == end(td_offsets.at(r.stop_)) ||
              tdo.valid_from_ < i->valid_from_) {
            i = td_offsets.at(r.stop_).insert(i, tdo);
          } else if (tdo.duration_ < i->duration_) {
            *i = tdo;
          }

          if (i + 1 == end(td_offsets.at(r.stop_)) ||
              (i + 1)->valid_from_ != tdo.valid_from_ + 1min) {
            td_offsets.at(r.stop_).insert(
                i + 1, {.valid_from_ = tdo.valid_from_ + 1min,
                        .duration_ = nigiri::footpath::kMaxDuration,
                        .transport_mode_id_ = mode});
          }
        }
      });

  for (auto& [l, tdos] : td_offsets) {
    for (auto i = begin(tdos); i != end(tdos);) {
      if (begin(tdos) < i && (i - 1)->duration_ == i->duration_ &&
          (i - 1)->transport_mode_id_ == i->transport_mode_id_) {
        i = tdos.erase(i);
      } else {
        ++i;
      }
    }
  }

  return td_offsets;
}

std::pair<nigiri::routing::td_offsets_t, nigiri::routing::td_offsets_t>
get_td_offsets_split(std::vector<nigiri::routing::offset> const&,
                     std::vector<service_times_t> const&,
                     nigiri::transport_mode_id_t);

}  // namespace motis::odm