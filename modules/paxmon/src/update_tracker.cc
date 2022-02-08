#include "motis/paxmon/update_tracker.h"

#include <cstdint>
#include <vector>

#include "utl/get_or_create.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "motis/hash_map.h"
#include "motis/hash_set.h"

#include "motis/core/access/trip_access.h"

#include "motis/paxmon/load_info.h"
#include "motis/paxmon/messages.h"
#include "motis/paxmon/universe.h"

using namespace flatbuffers;

namespace motis::paxmon {

struct update_tracker::impl {
  struct pg_base_info {
    data_source source_{};
    std::uint16_t pax_{};
    float probability_{};
    float previous_probability_{};
  };

  struct updated_trip_info {
    Offset<Vector<Offset<PaxMonEdgeLoadInfo>>> before_edges_{};
    mcd::hash_set<passenger_group_index> added_groups_;
    mcd::hash_set<passenger_group_index> removed_groups_;
    mcd::hash_set<passenger_group_index> reused_groups_;
  };

  impl(universe const& uv, schedule const& sched,
       bool const include_before_trip_load_info,
       bool const include_after_trip_load_info)
      : uv_{uv},
        sched_{sched},
        include_before_trip_load_info_{include_before_trip_load_info},
        include_after_trip_load_info_{include_after_trip_load_info} {}

  void before_group_added(passenger_group const* pg) {
    store_group_info(pg);
    added_groups_.emplace_back(pg->id_);
    for (auto& leg : pg->compact_planned_journey_.legs_) {
      auto& uti = get_or_create_updated_trip_info(leg.trip_idx_);
      uti.added_groups_.insert(pg->id_);
    }
  }

  void before_group_reused(passenger_group const* pg) {
    store_group_info(pg);
    reused_groups_.insert(pg->id_);
    // TODO(pablo): maybe use pg->edges_ instead
    for (auto& leg : pg->compact_planned_journey_.legs_) {
      auto& uti = get_or_create_updated_trip_info(leg.trip_idx_);
      uti.reused_groups_.insert(pg->id_);
    }
  }

  void after_group_reused(passenger_group const* pg) { store_group_info(pg); }

  void before_group_removed(passenger_group const* pg) {
    store_group_info(pg);
    removed_groups_.emplace_back(pg->id_);
    // TODO(pablo): maybe use pg->edges_ instead
    for (auto& leg : pg->compact_planned_journey_.legs_) {
      auto& uti = get_or_create_updated_trip_info(leg.trip_idx_);
      uti.removed_groups_.insert(pg->id_);
    }
  }

  std::pair<motis::module::message_creator&, Offset<PaxMonTrackedUpdates>>
  finish_updates() {
    auto const fb_updates = CreatePaxMonTrackedUpdates(
        mc_, added_groups_.size(), reused_groups_.size(),
        removed_groups_.size(), updated_trip_infos_.size(),
        mc_.CreateVector(utl::to_vec(updated_trip_infos_, [&](auto& entry) {
          return get_fbs_updated_trip(entry.first, entry.second);
        })));
    return {mc_, fb_updates};
  }

private:
  updated_trip_info& get_or_create_updated_trip_info(trip_idx_t const ti) {
    return utl::get_or_create(updated_trip_infos_, ti, [&]() {
      auto uti = updated_trip_info{};
      uti.before_edges_ = include_before_trip_load_info_
                              ? get_fbs_trip_load_info(ti)
                              : get_empty_fbs_trip_load_info();
      return uti;
    });
  }

  void store_group_info(passenger_group const* pg) {
    if (auto it = group_infos_.find(pg->id_); it != end(group_infos_)) {
      auto& pgbi = it->second;
      pgbi.probability_ = pg->probability_;
    } else {
      group_infos_[pg->id_] = pg_base_info{pg->source_, pg->passengers_,
                                           pg->probability_, pg->probability_};
    }
  }

  Offset<Vector<Offset<PaxMonEdgeLoadInfo>>> get_fbs_trip_load_info(
      trip_idx_t const ti) {
    auto const tli = calc_trip_load_info(uv_, get_trip(sched_, ti));
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
    auto removed_max_pax = 0U;
    auto removed_mean_pax = 0.F;
    auto added_max_pax = 0U;
    auto added_mean_pax = 0.F;

    mcd::hash_set<passenger_group_index> added_groups;
    for (auto const& pgi : uti.added_groups_) {
      auto const& pgbi = group_infos_.at(pgi);
      added_max_pax += pgbi.pax_;
      added_mean_pax += pgbi.probability_ * pgbi.pax_;
      added_groups.insert(pgi);
    }
    for (auto const& pgi : uti.reused_groups_) {
      // groups may be added first, then reused later, in that case all
      // pax have already been added above
      if (added_groups.find(pgi) != end(added_groups)) {
        continue;
      }
      auto const& pgbi = group_infos_.at(pgi);
      added_max_pax += pgbi.pax_;
      added_mean_pax +=
          (pgbi.probability_ - pgbi.previous_probability_) * pgbi.pax_;
      added_groups.insert(pgi);
    }
    for (auto const& pgi : uti.removed_groups_) {
      auto const& pgbi = group_infos_.at(pgi);
      removed_max_pax += pgbi.pax_;
      removed_mean_pax += pgbi.probability_ * pgbi.pax_;
    }

    return CreatePaxMonUpdatedTrip(
        mc_, to_fbs_trip_service_info(mc_, sched_, trp), removed_max_pax,
        removed_mean_pax, added_max_pax, added_mean_pax,
        mc_.CreateVectorOfStructs(
            utl::to_vec(uti.removed_groups_,
                        [&](passenger_group_index const pgi) {
                          return get_fbs_group_base_info(pgi);
                        })),
        mc_.CreateVectorOfStructs(
            utl::to_vec(uti.added_groups_,
                        [&](passenger_group_index const pgi) {
                          return get_fbs_group_base_info(pgi);
                        })),
        mc_.CreateVectorOfStructs(
            utl::to_vec(uti.reused_groups_,
                        [&](passenger_group_index const pgi) {
                          return get_fbs_reused_group_base_info(pgi);
                        })),
        uti.before_edges_,
        include_after_trip_load_info_ ? get_fbs_trip_load_info(ti)
                                      : get_empty_fbs_trip_load_info());
  }

  PaxMonGroupBaseInfo get_fbs_group_base_info(passenger_group_index const pgi) {
    auto const& pgbi = group_infos_.at(pgi);
    return PaxMonGroupBaseInfo{pgi, pgbi.pax_, pgbi.probability_};
  }

  PaxMonReusedGroupBaseInfo get_fbs_reused_group_base_info(
      passenger_group_index const pgi) {
    auto const& pgbi = group_infos_.at(pgi);
    return PaxMonReusedGroupBaseInfo{pgi, pgbi.pax_, pgbi.probability_,
                                     pgbi.previous_probability_};
  }

  universe const& uv_;
  schedule const& sched_;
  motis::module::message_creator mc_;
  bool include_before_trip_load_info_{};
  bool include_after_trip_load_info_{};

  mcd::hash_map<passenger_group_index, pg_base_info> group_infos_;
  std::vector<passenger_group_index> added_groups_;
  mcd::hash_set<passenger_group_index> reused_groups_;
  std::vector<passenger_group_index> removed_groups_;

  mcd::hash_map<trip_idx_t, updated_trip_info> updated_trip_infos_;
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

void update_tracker::start_tracking(universe const& uv, schedule const& sched,
                                    bool const include_before_trip_load_info,
                                    bool const include_after_trip_load_info) {
  utl::verify(!is_tracking(), "paxmon::update_tracker: already tracking");
  impl_ = std::make_unique<impl>(uv, sched, include_before_trip_load_info,
                                 include_after_trip_load_info);
}

std::pair<motis::module::message_creator&, Offset<PaxMonTrackedUpdates>>
update_tracker::finish_updates() {
  utl::verify(is_tracking(), "paxmon::update_tracker: not tracking");
  return impl_->finish_updates();
}

void update_tracker::stop_tracking() { impl_.reset(); }

bool update_tracker::is_tracking() const { return impl_ != nullptr; }

void update_tracker::before_group_added(passenger_group const* pg) {
  if (impl_) {
    impl_->before_group_added(pg);
  }
}

void update_tracker::before_group_reused(passenger_group const* pg) {
  if (impl_) {
    impl_->before_group_reused(pg);
  }
}

void update_tracker::after_group_reused(passenger_group const* pg) {
  if (impl_) {
    impl_->after_group_reused(pg);
  }
}

void update_tracker::before_group_removed(passenger_group const* pg) {
  if (impl_) {
    impl_->before_group_removed(pg);
  }
}

}  // namespace motis::paxmon
