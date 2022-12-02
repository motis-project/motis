#include "motis/paxforecast/measures/affected_groups.h"

#include <algorithm>

#include "utl/overloaded.h"

#include "motis/core/access/trip_access.h"

#include "motis/paxmon/localization.h"
#include "motis/paxmon/reachability.h"

using namespace motis::paxmon;

namespace motis::paxforecast::measures {

bool is_recipient(schedule const& sched, universe const& /*uv*/,
                  passenger_group const& /*pg*/,
                  passenger_localization const& loc, recipients const& rcpts) {
  if (loc.in_trip()) {
    return std::any_of(
        begin(rcpts.trips_), end(rcpts.trips_),
        [&](auto const& et) { return loc.in_trip_ == get_trip(sched, et); });
  } else {
    auto const loc_station = loc.at_station_->index_;
    return std::any_of(begin(rcpts.stations_), end(rcpts.stations_),
                       [&](auto const& si) { return loc_station == si; });
  }
}

passenger_localization localize(schedule const& sched, universe& uv,
                                affected_groups_info& result,
                                passenger_group_with_route const& pgwr,
                                group_route const& gr, time const loc_time) {
  if (auto const it = result.localization_.find(pgwr);
      it != end(result.localization_)) {
    return it->second;
  } else {
    return localize(sched,
                    get_reachability(uv, uv.passenger_groups_.journey(
                                             gr.compact_journey_index_)),
                    loc_time);
  }
}

template <typename CheckFn>
void add_affected_group_routes(schedule const& sched, universe& uv,
                               affected_groups_info& result,
                               recipients const& rcpts, time const loc_time,
                               measure_variant const* m,
                               CheckFn const& check_fn) {
  // TODO(pablo): performance optimizations - currently localizing every group
  for (auto const* pg : uv.passenger_groups_) {
    if (pg == nullptr) {
      continue;
    }
    for (auto const& gr : uv.passenger_groups_.routes(pg->id_)) {
      if (gr.broken_ || gr.probability_ == 0.0F) {
        continue;
      }
      auto const pgwr =
          passenger_group_with_route{pg->id_, gr.local_group_route_index_};
      auto const loc = localize(sched, uv, result, pgwr, gr, loc_time);
      if (is_recipient(sched, uv, *pg, loc, rcpts) && check_fn(pg, loc)) {
        result.localization_[pgwr] = loc;
        result.measures_[pgwr].emplace_back(m);
      }
    }
  }
}

bool matches_destination(
    passenger_localization const& loc,
    std::vector<std::uint32_t> const& planned_destinations) {
  return std::any_of(
      begin(loc.remaining_interchanges_), end(loc.remaining_interchanges_),
      [&](auto const station_id) {
        return std::find(begin(planned_destinations), end(planned_destinations),
                         station_id) != end(planned_destinations);
      });
}

affected_groups_info get_affected_groups(schedule const& sched, universe& uv,
                                         time const loc_time,
                                         measure_set const& measures) {
  auto result = affected_groups_info{};
  for (auto const& mv : measures) {
    std::visit(
        utl::overloaded{
            [&](trip_recommendation const& m) {
              add_affected_group_routes(
                  sched, uv, result, m.recipients_, loc_time, &mv,
                  [&](passenger_group const*,
                      passenger_localization const& loc) {
                    return matches_destination(loc, m.planned_destinations_);
                  });
            },
            [&](trip_load_information const& m) {
              add_affected_group_routes(
                  sched, uv, result, m.recipients_, loc_time, &mv,
                  [](passenger_group const*, passenger_localization const&) {
                    return true;
                  });
            },
            [&](trip_load_recommendation const& m) {
              add_affected_group_routes(
                  sched, uv, result, m.recipients_, loc_time, &mv,
                  [&](passenger_group const*,
                      passenger_localization const& loc) {
                    return matches_destination(loc, m.planned_destinations_);
                  });
            },
            [&](rt_update const& m) {
              add_affected_group_routes(
                  sched, uv, result, m.recipients_, loc_time, &mv,
                  [](passenger_group const*, passenger_localization const&) {
                    return true;
                  });
            }},
        mv);
  }
  return result;
}

}  // namespace motis::paxforecast::measures
