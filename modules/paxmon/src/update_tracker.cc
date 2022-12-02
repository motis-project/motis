#include "motis/paxmon/update_tracker.h"

#include <cmath>
#include <cstdint>
#include <algorithm>
#include <tuple>
#include <vector>

#include "utl/erase_if.h"
#include "utl/get_or_create.h"
#include "utl/to_vec.h"
#include "utl/verify.h"
#include "utl/zip.h"

#include "motis/hash_map.h"
#include "motis/hash_set.h"
#include "motis/pair.h"

#include "motis/core/access/trip_access.h"

#include "motis/paxmon/index_types.h"
#include "motis/paxmon/load_info.h"
#include "motis/paxmon/messages.h"
#include "motis/paxmon/universe.h"

using namespace flatbuffers;

namespace motis::paxmon {

struct update_tracker::impl {
  struct pgr_base_info {
    data_source source_{};
    std::uint16_t pax_{};
    float probability_{};
    float previous_probability_{};
    bool new_route_{};
  };

  struct edge_info {
    pax_stats stats_{};
    std::uint16_t capacity_{};
    bool has_capacity_{};
    bool critical_{};
  };

  struct critical_trip_info {
    int critical_sections_{};
    int max_excess_pax_{};
    int cumulative_excess_pax_{};
    std::vector<edge_info> edge_infos_;
  };

  struct updated_trip_info {
    Offset<Vector<Offset<PaxMonEdgeLoadInfo>>> before_edges_{};
    mcd::hash_set<passenger_group_with_route> updated_group_routes_;
    critical_trip_info before_cti_;
    critical_trip_info after_cti_;
    int critical_sections_diff_{};
    int max_excess_pax_diff_{};
    int cumulative_excess_pax_diff_{};
    int newly_critical_sections_{};
    int no_longer_critical_sections_{};
    int max_pax_increase_{};
    int max_pax_decrease_{};
    bool rerouted_{};
  };

  impl(universe const& uv, schedule const& sched,
       bool const include_before_trip_load_info,
       bool const include_after_trip_load_info,
       bool const include_trips_with_unchanged_load)
      : uv_{uv},
        sched_{sched},
        include_before_trip_load_info_{include_before_trip_load_info},
        include_after_trip_load_info_{include_after_trip_load_info},
        include_trips_with_unchanged_load_{include_trips_with_unchanged_load} {}

  void after_group_route_updated(passenger_group_with_route const pgwr,
                                 float const previous_probability,
                                 float const new_probability,
                                 bool const new_route) {
    auto const& pg = uv_.passenger_groups_.group(pgwr.pg_);
    auto const pgrbi =
        pgr_base_info{pg.source_, pg.passengers_, previous_probability,
                      new_probability, new_route};
    if (auto it = group_route_infos_.insert(mcd::pair{pgwr, pgrbi});
        it.second) {
      // new entry
      auto const& gr = uv_.passenger_groups_.route(pgwr);
      auto const cj = uv_.passenger_groups_.journey(gr.compact_journey_index_);
      for (auto const& leg : cj.legs()) {
        auto& uti = get_updated_trip_info(leg.trip_idx_);
        uti.updated_group_routes_.insert(pgwr);
      }
    } else {
      it.first->second.probability_ = new_probability;
    }
  }

  void before_trip_load_updated(trip_idx_t const ti) {
    get_or_create_updated_trip_info(ti);
  }

  void before_trip_rerouted(trip const* trp) {
    auto& uti = get_or_create_updated_trip_info(trp->trip_idx_);
    uti.rerouted_ = true;
  }

  std::pair<motis::module::message_creator&, Offset<PaxMonTrackedUpdates>>
  finish_updates() {
    finish_trips();

    auto sorted_trips = utl::to_vec(updated_trip_infos_, [](auto const& entry) {
      return std::pair{entry.first, &entry.second};
    });

    if (!include_trips_with_unchanged_load_) {
      utl::erase_if(sorted_trips, [](auto const& entry) {
        updated_trip_info const* uti = entry.second;
        return !uti->rerouted_ && uti->newly_critical_sections_ == 0 &&
               uti->no_longer_critical_sections_ == 0 &&
               uti->max_pax_increase_ == 0 && uti->max_pax_decrease_ == 0;
      });
    }

    std::sort(begin(sorted_trips), end(sorted_trips),
              [](auto const& lhs, auto const& rhs) {
                updated_trip_info const* l = lhs.second;
                updated_trip_info const* r = rhs.second;
                auto const crit_change_l = l->newly_critical_sections_ +
                                           l->no_longer_critical_sections_;
                auto const crit_change_r = r->newly_critical_sections_ +
                                           r->no_longer_critical_sections_;
                auto const max_change_l =
                    std::max(l->max_pax_increase_, l->max_pax_decrease_);
                auto const max_change_r =
                    std::max(r->max_pax_increase_, r->max_pax_decrease_);
                return std::tie(l->rerouted_, crit_change_l, max_change_l,
                                l->max_excess_pax_diff_) >
                       std::tie(r->rerouted_, crit_change_r, max_change_r,
                                r->max_excess_pax_diff_);
              });

    auto sorted_pgr_infos =
        utl::to_vec(group_route_infos_, [](auto const& entry) {
          return std::pair{entry.first, &entry.second};
        });
    std::sort(begin(sorted_pgr_infos), end(sorted_pgr_infos),
              [](auto const& lhs, auto const& rhs) {
                auto const& lhs_key = lhs.first;
                auto const& rhs_key = rhs.first;
                return std::tie(lhs_key.pg_, lhs_key.route_) <
                       std::tie(rhs_key.pg_, rhs_key.route_);
              });

    auto const fb_updates = CreatePaxMonTrackedUpdates(
        mc_, group_route_infos_.size(), updated_trip_infos_.size(),
        mc_.CreateVector(utl::to_vec(sorted_trips,
                                     [&](auto& entry) {
                                       return get_fbs_updated_trip(
                                           entry.first, *entry.second);
                                     })),
        mc_.CreateVectorOfStructs(
            utl::to_vec(sorted_pgr_infos, [](auto const& entry) {
              auto const& key = entry.first;
              auto const& bi = entry.second;
              return PaxMonGroupRouteUpdateInfo{key.pg_, key.route_, bi->pax_,
                                                bi->probability_,
                                                bi->previous_probability_};
            })));
    return {mc_, fb_updates};
  }

  void rt_updates_applied(tick_statistics const& tick_stats) {
    tick_stats_.emplace_back(tick_stats);
  }

private:
  updated_trip_info& get_or_create_updated_trip_info(trip_idx_t const ti) {
    return utl::get_or_create(updated_trip_infos_, ti, [&]() {
      auto uti = updated_trip_info{};
      auto const tli = calc_trip_load_info(uv_, get_trip(sched_, ti));
      uti.before_edges_ = include_before_trip_load_info_
                              ? get_fbs_trip_load_info(tli)
                              : get_empty_fbs_trip_load_info();
      uti.before_cti_ = get_critical_trip_info(tli);
      return uti;
    });
  }

  updated_trip_info& get_updated_trip_info(trip_idx_t const ti) {
    return updated_trip_infos_.at(ti);
  }

  void finish_trips() {
    for (auto& [ti, uti] : updated_trip_infos_) {
      uti.after_cti_ = get_critical_trip_info(
          calc_trip_load_info(uv_, get_trip(sched_, ti)));
      uti.critical_sections_diff_ =
          std::abs(uti.before_cti_.critical_sections_ -
                   uti.after_cti_.critical_sections_);
      uti.max_excess_pax_diff_ = std::abs(uti.before_cti_.max_excess_pax_ -
                                          uti.after_cti_.max_excess_pax_);
      uti.cumulative_excess_pax_diff_ =
          std::abs(uti.before_cti_.cumulative_excess_pax_ -
                   uti.after_cti_.cumulative_excess_pax_);

      if (!uti.rerouted_) {
        utl::verify(uti.before_cti_.edge_infos_.size() ==
                        uti.after_cti_.edge_infos_.size(),
                    "paxmon::update_tracker: edge count mismatch");
        for (auto const& [before, after] : utl::zip(
                 uti.before_cti_.edge_infos_, uti.after_cti_.edge_infos_)) {
          auto const pax_before = static_cast<int>(before.stats_.q95_);
          auto const pax_after = static_cast<int>(after.stats_.q95_);
          if (pax_before > pax_after) {
            uti.max_pax_decrease_ =
                std::max(uti.max_pax_decrease_, pax_before - pax_after);
          } else if (pax_before < pax_after) {
            uti.max_pax_increase_ =
                std::max(uti.max_pax_increase_, pax_after - pax_before);
          }
          if (before.critical_ && !after.critical_) {
            ++uti.no_longer_critical_sections_;
          } else if (!before.critical_ && after.critical_) {
            ++uti.newly_critical_sections_;
          }
        }
      } else {
        // just a guess (sections are not matched for rerouted trips)
        auto const count_crit = [](std::vector<edge_info> const& eis) {
          return std::count_if(begin(eis), end(eis), [](edge_info const& ei) {
            return ei.critical_;
          });
        };
        auto const crit_before = count_crit(uti.before_cti_.edge_infos_);
        auto const crit_after = count_crit(uti.after_cti_.edge_infos_);
        if (crit_before > crit_after) {
          uti.no_longer_critical_sections_ = crit_before - crit_after;
        } else if (crit_after > crit_before) {
          uti.newly_critical_sections_ = crit_after - crit_before;
        }
      }
    }
  }

  static critical_trip_info get_critical_trip_info(trip_load_info const& tli) {
    critical_trip_info cti;
    cti.critical_sections_ = 0;
    cti.cumulative_excess_pax_ = 0;
    cti.max_excess_pax_ = 0;
    cti.edge_infos_.reserve(tli.edges_.size());
    for (auto const& eli : tli.edges_) {
      cti.edge_infos_.emplace_back(
          edge_info{get_pax_stats(eli.forecast_cdf_), eli.edge_->capacity(),
                    eli.edge_->has_capacity(), eli.possibly_over_capacity_});
      if (!eli.edge_->has_capacity()) {
        continue;
      }
      if (eli.possibly_over_capacity_) {
        ++cti.critical_sections_;
      }
      auto const capacity = eli.edge_->capacity();
      auto const max_pax = eli.forecast_cdf_.data_.size();
      if (max_pax > capacity) {
        auto const excess_pax = static_cast<int>(max_pax - capacity);
        cti.cumulative_excess_pax_ += excess_pax;
        cti.max_excess_pax_ = std::max(cti.max_excess_pax_, excess_pax);
      }
    }
    return cti;
  }

  Offset<Vector<Offset<PaxMonEdgeLoadInfo>>> get_fbs_trip_load_info(
      trip_idx_t const ti) {
    return get_fbs_trip_load_info(
        calc_trip_load_info(uv_, get_trip(sched_, ti)));
  }

  Offset<Vector<Offset<PaxMonEdgeLoadInfo>>> get_fbs_trip_load_info(
      trip_load_info const& tli) {
    return mc_.CreateVector(utl::to_vec(tli.edges_, [&](auto const& eli) {
      return to_fbs(mc_, sched_, uv_, eli);
    }));
  }

  Offset<Vector<Offset<PaxMonEdgeLoadInfo>>> get_empty_fbs_trip_load_info() {
    return mc_.CreateVector(std::vector<Offset<PaxMonEdgeLoadInfo>>{});
  }

  Offset<PaxMonUpdatedTrip> get_fbs_updated_trip(trip_idx_t const ti,
                                                 updated_trip_info const& uti) {
    auto const* trp = get_trip(sched_, ti);

    return CreatePaxMonUpdatedTrip(
        mc_, to_fbs_trip_service_info(mc_, sched_, trp), uti.rerouted_,
        uti.newly_critical_sections_, uti.no_longer_critical_sections_,
        uti.max_pax_increase_, uti.max_pax_decrease_,
        get_fbs_critical_trip_info(uti.before_cti_),
        get_fbs_critical_trip_info(uti.after_cti_),
        mc_.CreateVectorOfStructs(
            utl::to_vec(uti.updated_group_routes_,
                        [](passenger_group_with_route const& pgwr) {
                          return PaxMonGroupWithRouteId{pgwr.pg_, pgwr.route_};
                        })),

        uti.before_edges_,
        include_after_trip_load_info_ ? get_fbs_trip_load_info(ti)
                                      : get_empty_fbs_trip_load_info());
  }

  Offset<PaxMonCriticalTripInfo> get_fbs_critical_trip_info(
      critical_trip_info const& cti) {
    return CreatePaxMonCriticalTripInfo(mc_, cti.critical_sections_,
                                        cti.max_excess_pax_,
                                        cti.cumulative_excess_pax_);
  }

  universe const& uv_;
  schedule const& sched_;
  motis::module::message_creator mc_;
  bool include_before_trip_load_info_{};
  bool include_after_trip_load_info_{};
  bool include_trips_with_unchanged_load_{};

  mcd::hash_map<passenger_group_with_route, pgr_base_info> group_route_infos_;
  mcd::hash_map<trip_idx_t, updated_trip_info> updated_trip_infos_;

public:
  std::vector<tick_statistics> tick_stats_;
};

update_tracker::update_tracker() = default;

update_tracker::update_tracker(update_tracker const&) {}

update_tracker::update_tracker(update_tracker&& o) noexcept
    : impl_{std::exchange(o.impl_, nullptr)} {}

update_tracker::~update_tracker() = default;

// NOLINTNEXTLINE(bugprone-unhandled-self-assignment, cert-oop54-cpp)
update_tracker& update_tracker::operator=(update_tracker const&) {
  return *this;
}

update_tracker& update_tracker::operator=(update_tracker&& o) noexcept {
  std::swap(impl_, o.impl_);
  return *this;
}

void update_tracker::start_tracking(
    universe const& uv, schedule const& sched,
    bool const include_before_trip_load_info,
    bool const include_after_trip_load_info,
    bool const include_trips_with_unchanged_load) {
  utl::verify(!is_tracking(), "paxmon::update_tracker: already tracking");
  impl_ = std::make_unique<impl>(uv, sched, include_before_trip_load_info,
                                 include_after_trip_load_info,
                                 include_trips_with_unchanged_load);
}

std::pair<motis::module::message_creator&, Offset<PaxMonTrackedUpdates>>
update_tracker::finish_updates() {
  utl::verify(is_tracking(), "paxmon::update_tracker: not tracking");
  return impl_->finish_updates();
}

void update_tracker::stop_tracking() { impl_.reset(); }

bool update_tracker::is_tracking() const { return impl_ != nullptr; }

std::vector<tick_statistics> update_tracker::get_tick_statistics() const {
  if (impl_) {
    return impl_->tick_stats_;
  } else {
    return {};
  }
}

void update_tracker::after_group_route_updated(
    passenger_group_with_route const pgwr, float const previous_probability,
    float const new_probability, bool const new_route) {
  if (impl_) {
    impl_->after_group_route_updated(pgwr, previous_probability,
                                     new_probability, new_route);
  }
}

void update_tracker::before_trip_load_updated(trip_idx_t const ti) {
  if (impl_) {
    impl_->before_trip_load_updated(ti);
  }
}

void update_tracker::before_trip_rerouted(trip const* trp) {
  if (impl_) {
    impl_->before_trip_rerouted(trp);
  }
}

void update_tracker::rt_updates_applied(tick_statistics const& tick_stats) {
  if (impl_) {
    impl_->rt_updates_applied(tick_stats);
  }
}

}  // namespace motis::paxmon
