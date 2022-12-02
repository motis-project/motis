#include "motis/paxmon/api/get_interchanges.h"

#include <algorithm>

#include "utl/enumerate.h"
#include "utl/pipes.h"
#include "utl/to_vec.h"

#include "motis/hash_map.h"

#include "motis/core/access/trip_access.h"
#include "motis/core/conv/station_conv.h"
#include "motis/core/conv/trip_conv.h"

#include "motis/paxmon/compact_journey_util.h"
#include "motis/paxmon/get_load.h"
#include "motis/paxmon/get_universe.h"
#include "motis/paxmon/messages.h"

using namespace motis::module;
using namespace motis::paxmon;

namespace motis::paxmon::api {

struct grouped_key {
  std::uint32_t group_station_{};
  std::uint32_t entry_station_{};
  time entry_time_{};
  trip const* other_trp_{};
};

std::pair<std::uint32_t /*station*/, time> get_group_route_entry(
    universe const& uv, schedule const& sched, group_route const& gr,
    trip_idx_t const ti) {
  for (auto const& ei : uv.passenger_groups_.route_edges(gr.edges_index_)) {
    auto const* e = ei.get(uv);
    for (auto const& trp : e->get_trips(sched)) {
      if (trp->trip_idx_ == ti) {
        auto const* from = e->from(uv);
        return {from->station_idx(), from->schedule_time()};
      }
    }
  }
  return {0, 0};
}

trip const* get_trip_before_entry(schedule const& sched, edge const* e,
                                  fws_compact_journey const& cj) {
  auto const& merged_trips = e->get_trips(sched);
  for (auto const& leg_with_index : utl::enumerate(cj.legs())) {
    auto const leg_idx = std::get<0>(leg_with_index);
    auto const& leg = std::get<1>(leg_with_index);
    if (std::find_if(begin(merged_trips), end(merged_trips),
                     [&](trip const* t) {
                       return t->trip_idx_ == leg.trip_idx_;
                     }) != end(merged_trips)) {
      if (leg_idx > 0) {
        return get_trip(sched, cj.legs().at(leg_idx - 1).trip_idx_);
      } else {
        break;
      }
    }
  }
  return nullptr;
}

motis::module::msg_ptr get_groups_in_trip(paxmon_data& data,
                                          motis::module::msg_ptr const& msg) {
  auto const req = motis_content(PaxMonGetGroupsInTripRequest, msg);
  auto const uv_access = get_universe_and_schedule(data, req->universe());
  auto const& sched = uv_access.sched_;
  auto& uv = uv_access.uv_;
  auto const trp = from_fbs(sched, req->trip());
  auto const grp_filter = req->filter();
  auto const grp_by_station = req->group_by_station();
  auto const grp_by_other_trip = req->group_by_other_trip();
  auto const include_group_infos = req->include_group_infos();

  auto const get_key = [&](group_route const& gr, fws_compact_journey const& cj,
                           trip const* other_trip) {
    auto key = grouped_key{};
    if (grp_by_other_trip) {
      key.other_trp_ = other_trip;
    }
    switch (grp_by_station) {
      case PaxMonGroupByStation_First:
        key.group_station_ = cj.start_station_id();
        break;
      case PaxMonGroupByStation_Last:
        key.group_station_ = cj.destination_station_id();
        break;
      case PaxMonGroupByStation_FirstLongDistance:
        key.group_station_ =
            get_first_long_distance_station_id(uv, cj).value_or(
                cj.start_station_id());
        break;
      case PaxMonGroupByStation_LastLongDistance:
        key.group_station_ = get_last_long_distance_station_id(uv, cj).value_or(
            cj.destination_station_id());
        break;
      case PaxMonGroupByStation_EntryAndLast: {
        key.group_station_ = cj.destination_station_id();
        auto const [entry_station, entry_time] =
            get_group_route_entry(uv, sched, gr, trp->trip_idx_);
        key.entry_station_ = entry_station;
        key.entry_time_ = entry_time;
        break;
      }
      default: break;
    }
    return key;
  };

  auto const group_route_enters_here =
      [&](event_node const* trp_node, edge const* e,
          fws_compact_journey const& cj) -> std::pair<bool, trip const*> {
    auto const& merged_trips = e->get_trips(sched);
    for (auto const& leg_with_index : utl::enumerate(cj.legs())) {
      auto const leg_idx = std::get<0>(leg_with_index);
      auto const& leg = std::get<1>(leg_with_index);
      if (leg.enter_station_id_ == trp_node->station_idx() &&
          std::find_if(begin(merged_trips), end(merged_trips),
                       [&](trip const* t) {
                         return t->trip_idx_ == leg.trip_idx_;
                       }) != end(merged_trips)) {
        return {true, leg_idx > 0
                          ? get_trip(sched, cj.legs().at(leg_idx - 1).trip_idx_)
                          : nullptr};
      }
    }
    return {false, nullptr};
  };

  auto const group_route_exits_here =
      [&](event_node const* trp_node, edge const* e,
          fws_compact_journey const& cj) -> std::pair<bool, trip const*> {
    auto const& merged_trips = e->get_trips(sched);
    for (auto const& leg_with_index : utl::enumerate(cj.legs())) {
      auto const leg_idx = std::get<0>(leg_with_index);
      auto const& leg = std::get<1>(leg_with_index);
      if (leg.exit_station_id_ == trp_node->station_idx() &&
          std::find_if(begin(merged_trips), end(merged_trips),
                       [&](trip const* t) {
                         return t->trip_idx_ == leg.trip_idx_;
                       }) != end(merged_trips)) {
        return {true, leg_idx < cj.legs().size() - 1
                          ? get_trip(sched, cj.legs().at(leg_idx + 1).trip_idx_)
                          : nullptr};
      }
    }
    return {false, nullptr};
  };

  message_creator mc;

  auto const make_section_info = [&](edge const* e) {
    auto const* from = e->from(uv);
    auto const* to = e->to(uv);

    struct grouped_pgwrs_t {
      std::vector<passenger_group_with_route> group_routes_{};
      std::uint32_t min_pax_{};
      std::uint32_t max_pax_{};
      float avg_pax_{};
      pax_pdf pdf_{};
      pax_stats pax_stats_{};
    };

    mcd::hash_map<grouped_key, grouped_pgwrs_t> grouped;
    std::vector<flatbuffers::Offset<GroupedPassengerGroups>> grouped_pgs_vec;

    for (auto const& pgwr : uv.pax_connection_info_.group_routes(e->pci_)) {
      auto const& pg = uv.passenger_groups_.group(pgwr.pg_);
      auto const& gr = uv.passenger_groups_.route(pgwr);
      if (gr.probability_ == 0.0F) {
        continue;
      }
      trip const* other_trp = nullptr;
      auto const cj = uv.passenger_groups_.journey(gr.compact_journey_index_);

      if (grp_filter == PaxMonGroupFilter_Entering) {
        auto const [entering, ot] = group_route_enters_here(from, e, cj);
        if (entering) {
          other_trp = ot;
        } else {
          continue;
        }
      } else if (grp_filter == PaxMonGroupFilter_Exiting) {
        auto const [exiting, ot] = group_route_exits_here(to, e, cj);
        if (exiting) {
          other_trp = ot;
        } else {
          continue;
        }
      }

      if (grp_by_other_trip &&
          grp_by_station == PaxMonGroupByStation_EntryAndLast) {
        other_trp = get_trip_before_entry(sched, e, cj);
      }

      auto const key = get_key(gr, cj, other_trp);
      auto& gg = grouped[key];
      gg.group_routes_.emplace_back(pgwr);
      gg.max_pax_ += pg.passengers_;
      if (gr.probability_ == 1.0F) {
        gg.min_pax_ += pg.passengers_;
      }
      gg.avg_pax_ += pg.passengers_ * gr.probability_;
    }

    for (auto& [key, gbd] : grouped) {
      gbd.pdf_ = get_load_pdf(uv.passenger_groups_, gbd.group_routes_);
      gbd.pax_stats_ = get_pax_stats(get_cdf(gbd.pdf_));
    }

    auto sorted_keys =
        utl::to_vec(grouped, [](auto const& kv) { return kv.first; });
    std::sort(begin(sorted_keys), end(sorted_keys),
              [&](auto const& a, auto const& b) {
                return grouped[a].pax_stats_.q50_ > grouped[b].pax_stats_.q50_;
              });
    for (auto const& key : sorted_keys) {
      auto& gbd = grouped[key];
      auto const grouped_by_station =
          key.group_station_ != 0
              ? mc.CreateVector(std::vector<flatbuffers::Offset<Station>>{
                    to_fbs(mc, *sched.stations_[key.group_station_])})
              : mc.CreateVector(
                    static_cast<flatbuffers::Offset<Station>*>(nullptr), 0);
      auto const grouped_by_trip =
          key.other_trp_ != nullptr
              ? mc.CreateVector(
                    std::vector<flatbuffers::Offset<TripServiceInfo>>{
                        to_fbs_trip_service_info(mc, sched, key.other_trp_)})
              : mc.CreateVector(
                    static_cast<flatbuffers::Offset<TripServiceInfo>*>(nullptr),
                    0);
      auto const entry_station =
          key.entry_station_ != 0
              ? mc.CreateVector(std::vector<flatbuffers::Offset<Station>>{
                    to_fbs(mc, *sched.stations_[key.entry_station_])})
              : mc.CreateVector(
                    static_cast<flatbuffers::Offset<Station>*>(nullptr), 0);
      auto const entry_time =
          key.entry_time_ != 0 ? motis_to_unixtime(sched, key.entry_time_) : 0;

      if (include_group_infos) {
        std::sort(begin(gbd.group_routes_), end(gbd.group_routes_));
      }

      grouped_pgs_vec.emplace_back(CreateGroupedPassengerGroups(
          mc, grouped_by_station, grouped_by_trip, entry_station, entry_time,
          CreatePaxMonCombinedGroupRoutes(
              mc,
              mc.CreateVectorOfStructs(
                  include_group_infos
                      ? utl::to_vec(
                            gbd.group_routes_,
                            [&](passenger_group_with_route const& pgwr) {
                              return to_fbs_base_info(mc, uv.passenger_groups_,
                                                      pgwr);
                            })
                      : std::vector<PaxMonGroupRouteBaseInfo>{}),
              to_fbs_distribution(mc, gbd.pdf_, gbd.pax_stats_))));
    }

    return CreateGroupsInTripSection(
        mc, to_fbs(mc, from->get_station(sched)),
        to_fbs(mc, to->get_station(sched)),
        motis_to_unixtime(sched, from->schedule_time()),
        motis_to_unixtime(sched, from->current_time()),
        motis_to_unixtime(sched, to->schedule_time()),
        motis_to_unixtime(sched, to->current_time()),
        mc.CreateVector(grouped_pgs_vec));
  };

  mc.create_and_finish(
      MsgContent_PaxMonGetGroupsInTripResponse,
      CreatePaxMonGetGroupsInTripResponse(
          mc,
          mc.CreateVector(
              utl::all(uv.trip_data_.edges(trp))  //
              | utl::transform([&](auto const e) { return e.get(uv); })  //
              | utl::remove_if([](auto const* e) { return !e->is_trip(); })  //
              | utl::transform(
                    [&](auto const* e) { return make_section_info(e); })  //
              | utl::vec()))
          .Union());
  return make_msg(mc);
}

}  // namespace motis::paxmon::api
