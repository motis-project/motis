#include "motis/paxmon/api/filter_groups.h"

#include <cassert>
#include <cstdint>
#include <algorithm>
#include <iterator>
#include <limits>
#include <string_view>
#include <tuple>

#include "utl/to_vec.h"

#include "motis/hash_set.h"

#include "motis/core/access/station_access.h"

#include "motis/paxmon/get_load.h"
#include "motis/paxmon/get_universe.h"
#include "motis/paxmon/messages.h"

using namespace motis::module;
using namespace motis::paxmon;
using namespace flatbuffers;

namespace motis::paxmon::api {

namespace {

struct group_info {
  passenger_group_index pgi_{};

  time scheduled_departure_{INVALID_TIME};

  std::int16_t min_estimated_delay_{std::numeric_limits<std::int16_t>::max()};
  std::int16_t max_estimated_delay_{std::numeric_limits<std::int16_t>::min()};
  float expected_estimated_delay_{};
  float p_destination_unreachable_{};
  std::uint32_t log_entries_{};
};

mcd::hash_set<std::uint32_t> get_stations(
    schedule const& sched, Vector<Offset<String>> const* fbs_vec) {
  mcd::hash_set<std::uint32_t> station_ids;
  for (auto const& eva : *fbs_vec) {
    auto const* st =
        get_station(sched, std::string_view{eva->data(), eva->size()});
    station_ids.insert(st->index_);
    for (auto const& eq : st->equivalent_) {
      station_ids.insert(eq->index_);
    }
  }
  return station_ids;
}

std::vector<trip_idx_t> get_trips(schedule const& sched,
                                  Vector<std::uint32_t> const* fbs_vec) {
  std::vector<trip_idx_t> trip_indices;
  for (auto const train_nr : *fbs_vec) {
    auto const search_entry = std::make_pair(primary_trip_id{0U, train_nr, 0U},
                                             static_cast<trip*>(nullptr));
    for (auto it = std::lower_bound(begin(sched.trips_), end(sched.trips_),
                                    search_entry);
         it != end(sched.trips_) && it->first.train_nr_ == train_nr; ++it) {
      auto const trp = static_cast<trip const*>(it->second);
      trip_indices.emplace_back(trp->trip_idx_);
    }
  }
  return trip_indices;
}

}  // namespace

msg_ptr filter_groups(paxmon_data& data, msg_ptr const& msg) {
  auto const req = motis_content(PaxMonFilterGroupsRequest, msg);
  auto const uv_access = get_universe_and_schedule(data, req->universe());
  auto const& sched = uv_access.sched_;
  auto const& uv = uv_access.uv_;
  auto const& pgc = uv.passenger_groups_;

  auto const include_reroute_log = req->include_reroute_log();
  auto const max_results = req->max_results();
  auto const skip_first = req->skip_first();

  auto const filter_start_stations =
      get_stations(sched, req->filter_by_start());
  auto const filter_destination_stations =
      get_stations(sched, req->filter_by_destination());
  auto const filter_via_stations = get_stations(sched, req->filter_by_via());

  auto const filter_group_ids = utl::to_vec(*req->filter_by_group_id());
  auto const filter_sources =
      utl::to_vec(*req->filter_by_data_source(),
                  [](PaxMonDataSource const* ds) { return from_fbs(ds); });

  auto const filter_trip_indices = get_trips(sched, req->filter_by_train_nr());

  auto const filter_reroute_reasons = utl::to_vec(
      *req->filter_by_reroute_reason(),
      [](auto const& rr) { return static_cast<reroute_reason_t>(rr); });

  auto const filter_by_start = !filter_start_stations.empty();
  auto const filter_by_destination = !filter_destination_stations.empty();
  auto const filter_by_via = !filter_via_stations.empty();
  auto const filter_by_stations =
      filter_by_start || filter_by_destination || filter_by_via;
  auto const filter_by_journey = filter_by_stations;

  auto const filter_by_ids =
      !filter_group_ids.empty() || !filter_sources.empty();

  auto const filter_by_trips = req->filter_by_train_nr()->size() > 0;

  auto const time_filter_type = req->filter_by_time();
  auto const filter_by_time =
      time_filter_type != PaxMonFilterGroupsTimeFilter_NoFilter;
  auto const filter_interval_begin =
      unix_to_motistime(sched.schedule_begin_, req->filter_interval()->begin());
  auto const filter_interval_end =
      unix_to_motistime(sched.schedule_begin_, req->filter_interval()->end());

  auto const filter_by_reroute_reason = !filter_reroute_reasons.empty();

  std::vector<group_info> selected_groups;
  selected_groups.reserve(pgc.size());
  auto total_matching_passengers = 0ULL;

  auto check_trip_filter = [&](fws_compact_journey const cj) {
    return std::any_of(begin(cj.legs()), end(cj.legs()), [&](auto const& leg) {
      return std::find(begin(filter_trip_indices), end(filter_trip_indices),
                       leg.trip_idx_) != end(filter_trip_indices);
    });
  };

  auto check_time_filter = [&](fws_compact_journey const cj) {
    auto const dep = cj.legs().front().enter_time_;
    auto const arr = cj.legs().back().exit_time_;
    if (time_filter_type == PaxMonFilterGroupsTimeFilter_DepartureTime) {
      return dep >= filter_interval_begin && dep < filter_interval_end;
    } else if (time_filter_type ==
               PaxMonFilterGroupsTimeFilter_DepartureOrArrivalTime) {
      return (dep >= filter_interval_begin && dep < filter_interval_end) ||
             (arr >= filter_interval_begin && arr < filter_interval_end);
    } else {
      return true;
    }
  };

  auto const handle_group = [&](passenger_group const* pg) {
    auto const pgi = pg->id_;
    auto gi = group_info{pgi};
    auto station_filter_match = !filter_by_journey;
    auto trip_filter_match = !filter_by_trips;
    auto time_filter_match = !filter_by_time;

    auto const log = pgc.reroute_log_entries(pgi);
    gi.log_entries_ = log.size();
    if (filter_by_reroute_reason) {
      if (!std::any_of(
              log.begin(), log.end(), [&](reroute_log_entry const& entry) {
                return std::find(filter_reroute_reasons.begin(),
                                 filter_reroute_reasons.end(),
                                 entry.reason_) != filter_reroute_reasons.end();
              })) {
        return;
      }
    }

    for (auto const& gr : pgc.routes(pgi)) {
      if (gr.probability_ != 0) {
        gi.min_estimated_delay_ =
            std::min(gi.min_estimated_delay_, gr.estimated_delay_);
        gi.max_estimated_delay_ =
            std::max(gi.max_estimated_delay_, gr.estimated_delay_);
        gi.expected_estimated_delay_ += gr.probability_ * gr.estimated_delay_;
        if (gr.destination_unreachable_) {
          gi.p_destination_unreachable_ += gr.probability_;
        }
      }
      if (gr.planned_ && gi.scheduled_departure_ == INVALID_TIME) {
        auto const cj = pgc.journey(gr.compact_journey_index_);
        assert(!cj.legs().empty());
        gi.scheduled_departure_ = cj.legs().front().enter_time_;
        if (!trip_filter_match) {
          trip_filter_match = check_trip_filter(cj);
        }
        if (!time_filter_match) {
          time_filter_match = check_time_filter(cj);
        }
        if (!station_filter_match) {
          // check station filter
          auto const start_matches =
              !filter_by_start || (filter_start_stations.find(
                                       cj.legs().front().enter_station_id_) !=
                                   filter_start_stations.end());
          if (!start_matches) {
            continue;
          }
          auto const destination_matches =
              !filter_by_destination ||
              (filter_destination_stations.find(
                   cj.legs().back().exit_station_id_) !=
               filter_destination_stations.end());
          if (!destination_matches) {
            continue;
          }
          auto const via_matches =
              !filter_by_via ||
              std::any_of(
                  begin(cj.legs()), end(cj.legs()), [&](auto const& leg) {
                    return filter_via_stations.find(leg.enter_station_id_) !=
                               filter_via_stations.end() ||
                           filter_via_stations.find(leg.exit_station_id_) !=
                               filter_via_stations.end();
                  });
          station_filter_match = via_matches;
        }
      } else if (!trip_filter_match || !time_filter_match) {
        auto const cj = pgc.journey(gr.compact_journey_index_);
        if (!trip_filter_match) {
          trip_filter_match = check_trip_filter(cj);
        }
        if (!time_filter_match) {
          time_filter_match = check_time_filter(cj);
        }
      }
    }

    if (station_filter_match && trip_filter_match && time_filter_match) {
      selected_groups.emplace_back(gi);
      total_matching_passengers += pg->passengers_;
    }
  };

  auto const add_by_data_source = [&](data_source const& ds) -> bool {
    if (auto const it = pgc.groups_by_source_.find(ds);
        it != end(pgc.groups_by_source_)) {
      for (auto const pgid : it->second) {
        if (auto const pg = pgc.at(pgid); pg != nullptr) {
          handle_group(pg);
        }
      }
      return true;
    }
    return false;
  };

  if (filter_by_ids) {
    for (auto const pgi : filter_group_ids) {
      if (auto const pg = uv.passenger_groups_.at(pgi); pg != nullptr) {
        handle_group(pg);
      }
    }

    for (auto ds : filter_sources) {
      if (ds.secondary_ref_ != 0) {
        add_by_data_source(ds);
      } else {
        ds.secondary_ref_ = 1;
        while (add_by_data_source(ds)) {
          ++ds.secondary_ref_;
        }
      }
    }
  } else {
    for (auto const& pg : pgc) {
      handle_group(pg);
    }
  }

  switch (req->sort_by()) {
    case PaxMonFilterGroupsSortOrder_GroupId:
      std::stable_sort(begin(selected_groups), end(selected_groups),
                       [](group_info const& lhs, group_info const& rhs) {
                         return lhs.pgi_ < rhs.pgi_;
                       });
      break;
    case PaxMonFilterGroupsSortOrder_ScheduledDepartureTime:
      std::stable_sort(begin(selected_groups), end(selected_groups),
                       [](group_info const& lhs, group_info const& rhs) {
                         return lhs.scheduled_departure_ <
                                rhs.scheduled_departure_;
                       });
      break;
    case PaxMonFilterGroupsSortOrder_MaxEstimatedDelay:
      std::stable_sort(begin(selected_groups), end(selected_groups),
                       [](group_info const& lhs, group_info const& rhs) {
                         return std::tie(lhs.max_estimated_delay_,
                                         lhs.expected_estimated_delay_) >
                                std::tie(rhs.max_estimated_delay_,
                                         rhs.expected_estimated_delay_);
                       });
      break;
    case PaxMonFilterGroupsSortOrder_ExpectedEstimatedDelay:
      std::stable_sort(begin(selected_groups), end(selected_groups),
                       [](group_info const& lhs, group_info const& rhs) {
                         return std::tie(lhs.expected_estimated_delay_,
                                         lhs.max_estimated_delay_) >
                                std::tie(rhs.expected_estimated_delay_,
                                         rhs.max_estimated_delay_);
                       });
      break;
    case PaxMonFilterGroupsSortOrder_MinEstimatedDelay:
      std::stable_sort(begin(selected_groups), end(selected_groups),
                       [](group_info const& lhs, group_info const& rhs) {
                         return std::tie(lhs.min_estimated_delay_,
                                         lhs.expected_estimated_delay_) <
                                std::tie(rhs.min_estimated_delay_,
                                         rhs.expected_estimated_delay_);
                       });
      break;
    case PaxMonFilterGroupsSortOrder_RerouteLogEntries:
      std::stable_sort(begin(selected_groups), end(selected_groups),
                       [](group_info const& lhs, group_info const& rhs) {
                         return lhs.log_entries_ > rhs.log_entries_;
                       });
      break;
    default: break;
  }

  auto const total_matching_groups = selected_groups.size();
  if (skip_first > 0) {
    selected_groups.erase(
        begin(selected_groups),
        std::next(begin(selected_groups),
                  std::min(static_cast<std::size_t>(skip_first),
                           selected_groups.size())));
  }

  auto remaining_groups = 0ULL;
  if (max_results != 0 && selected_groups.size() > max_results) {
    remaining_groups = selected_groups.size() - max_results;
    selected_groups.resize(max_results);
  }

  message_creator mc;
  mc.create_and_finish(MsgContent_PaxMonFilterGroupsResponse,
                       CreatePaxMonFilterGroupsResponse(
                           mc, total_matching_groups, total_matching_passengers,
                           selected_groups.size(), remaining_groups,
                           skip_first + selected_groups.size(),
                           mc.CreateVector(utl::to_vec(
                               selected_groups,
                               [&](group_info const& gi) {
                                 return CreatePaxMonGroupWithStats(
                                     mc,
                                     to_fbs(sched, pgc, mc, pgc.group(gi.pgi_),
                                            include_reroute_log),
                                     gi.min_estimated_delay_,
                                     gi.max_estimated_delay_,
                                     gi.expected_estimated_delay_,
                                     gi.p_destination_unreachable_);
                               })))
                           .Union());
  return make_msg(mc);
}

}  // namespace motis::paxmon::api
