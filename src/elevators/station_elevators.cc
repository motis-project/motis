#include "motis/elevators/station_elevators.h"

#include <map>
#include <optional>

#include "utl/helpers/algorithm.h"

#include "nigiri/logging.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/station_paths.h"
#include "nigiri/timetable.h"

#include "motis/elevators/elevators.h"

namespace n = nigiri;

namespace motis {

// Max. distance between a pathway elevator and a FaSta / SIRI-FM elevator.
constexpr auto const kStationElevatorRadius = 20.0;

void block_station_elevators(n::timetable const& tt,
                             n::station_paths const* sp,
                             elevators const& e,
                             n::rt_timetable& rtt) {
  for (auto& b : rtt.blocked_restricted_out_) {
    b = {};
  }
  for (auto& b : rtt.blocked_restricted_in_) {
    b = {};
  }
  rtt.blocked_entrance_paths_ = {};
  if (sp == nullptr || sp->pieces_.size() == 0U || e.elevators_.empty()) {
    return;
  }

  auto const& loc = tt.locations_;
  for (auto prf = n::profile_idx_t{0U}; prf != n::kNProfiles; ++prf) {
    rtt.blocked_restricted_out_[prf].resize(static_cast<n::bitvec::size_type>(
        loc.footpath_restrictions_out_[prf].data_.size()));
    rtt.blocked_restricted_in_[prf].resize(static_cast<n::bitvec::size_type>(
        loc.footpath_restrictions_in_[prf].data_.size()));
  }

  // Footpath (from, to, record): blocked if all its paths are blocked.
  auto const block = [&](n::location_idx_t const from,
                         n::location_idx_t const to,
                         n::footpath_restriction const& r) {
    for (auto prf = n::profile_idx_t{0U}; prf != n::kNProfiles; ++prf) {
      // restricted tail of l's list: record index = records start + i
      auto const mark = [&](bool const is_out, n::bitvec& blocked,
                            n::location_idx_t const l,
                            n::location_idx_t const target) {
        auto const& fps =
            (is_out ? loc.footpaths_out_ : loc.footpaths_in_)[prf];
        auto const& records = (is_out ? loc.footpath_restrictions_out_
                                      : loc.footpath_restrictions_in_)[prf];
        if (l >= fps.size() || l >= records.size()) {
          return;
        }
        auto const bucket = fps[l];
        auto const rs = records[l];
        auto const split = loc.restricted_begin(prf, is_out, l);
        auto const start = records.bucket_starts_[to_idx(l)];
        for (auto i = split; i != bucket.size(); ++i) {
          if (bucket[i].target() == target && rs[i - split] == r) {
            blocked.set(static_cast<n::bitvec::size_type>(start + i - split),
                        true);
          }
        }
      };
      loc.for_each_member_pair(
          from, to, [&](n::location_idx_t const a, n::location_idx_t const b) {
            mark(true, rtt.blocked_restricted_out_[prf], a, b);
            mark(false, rtt.blocked_restricted_in_[prf], b, a);
          });
    }
  };

  // Elevator pieces next to an elevator that is out of service.
  auto blocked_pieces = n::bitvec{};
  blocked_pieces.resize(static_cast<n::bitvec::size_type>(sp->pieces_.size()));
  auto n_blocked_pieces = 0U;
  for (auto i = 0U; i != sp->pieces_.size(); ++i) {
    auto const& p = sp->pieces_[n::station_piece_idx_t{i}];
    if (p.mode_ != n::station_piece_mode::kElevator) {
      continue;
    }
    auto const& a = sp->nodes_[p.from_];
    auto const& b = sp->nodes_[p.to_];
    auto const pos = a.has_pos()   ? std::optional{a.pos_}
                     : b.has_pos() ? std::optional{b.pos_}
                                   : std::nullopt;
    if (!pos.has_value()) {
      continue;
    }
    auto is_out = false;
    e.elevators_rtree_.in_radius(*pos, kStationElevatorRadius,
                                 [&](elevator_idx_t const x) {
                                   is_out = is_out || !e.elevators_[x].status_;
                                 });
    if (is_out) {
      blocked_pieces.set(i, true);
      ++n_blocked_pieces;
    }
  }
  if (n_blocked_pieces == 0U) {
    return;
  }
  auto const uses_blocked = [&](auto const& pieces) {
    return utl::any_of(pieces, [&](n::station_piece_idx_t const x) {
      return blocked_pieces.test(to_idx(x));
    });
  };

  auto n_blocked_paths = 0U;
  auto const& paths = sp->paths_;
  for (auto i = 0U; i != paths.size();) {
    auto const& first = paths[n::station_path_idx_t{i}];
    auto all_blocked =
        std::map<std::uint32_t, std::pair<bool, n::footpath_restriction>>{};
    auto j = i;
    for (; j != paths.size() &&
           paths[n::station_path_idx_t{j}].from_ == first.from_ &&
           paths[n::station_path_idx_t{j}].to_ == first.to_;
         ++j) {
      auto const& p = paths[n::station_path_idx_t{j}];
      auto const r = p.restriction_;
      auto const key = (std::uint32_t{r.flags_} << 24U) |
                       (std::uint32_t{r.slope_} << 16U) |
                       (std::uint32_t{r.width_} << 8U) | r.fixed_;
      auto const is_blocked =
          uses_blocked(sp->path_pieces_[n::station_path_idx_t{j}]);
      auto const [it, is_new] =
          all_blocked.emplace(key, std::pair{is_blocked, r});
      if (!is_new) {
        it->second.first = it->second.first && is_blocked;
      }
    }
    for (auto const& [_, x] : all_blocked) {
      if (x.first) {
        block(first.from_, first.to_, x.second);
        ++n_blocked_paths;
      }
    }
    i = j;
  }

  rtt.blocked_entrance_paths_.resize(
      static_cast<n::bitvec::size_type>(sp->entrance_paths_.size()));
  auto n_blocked_entrance_paths = 0U;
  for (auto i = 0U; i != sp->entrance_paths_.size(); ++i) {
    if (uses_blocked(
            sp->entrance_path_pieces_[n::station_entrance_path_idx_t{i}])) {
      rtt.blocked_entrance_paths_.set(i, true);
      ++n_blocked_entrance_paths;
    }
  }

  n::log(n::log_lvl::info, "motis.rt.elevators",
         "station elevators: {} pieces out of service, {} footpaths, {} "
         "entrance paths blocked",
         n_blocked_pieces, n_blocked_paths, n_blocked_entrance_paths);
}

}  // namespace motis
