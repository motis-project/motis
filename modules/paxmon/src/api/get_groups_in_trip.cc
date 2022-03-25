#include "motis/paxmon/api/get_interchanges.h"

#include <algorithm>

#include "utl/enumerate.h"
#include "utl/pipes.h"
#include "utl/to_vec.h"

#include "motis/hash_map.h"
#include "motis/pair.h"

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

std::pair<std::uint32_t /*station*/, time> get_group_entry(
    passenger_group const* pg, trip_idx_t const ti) {
  for (auto const& leg : pg->compact_planned_journey_.legs_) {
    if (leg.trip_idx_ == ti) {
      return {leg.enter_station_id_, leg.enter_time_};
    }
  }
  return {0, 0};
}

trip const* get_trip_before_entry(schedule const& sched,
                                  passenger_group const* pg,
                                  trip_idx_t const ti) {
  for (auto const& [leg_idx, leg] :
       utl::enumerate(pg->compact_planned_journey_.legs_)) {
    if (leg.trip_idx_ == ti) {
      if (leg_idx > 0) {
        return get_trip(
            sched, pg->compact_planned_journey_.legs_[leg_idx - 1].trip_idx_);
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

  auto const get_key = [&](passenger_group const* pg, trip const* other_trip) {
    auto key = grouped_key{};
    if (grp_by_other_trip) {
      key.other_trp_ = other_trip;
    }
    auto const& cj = pg->compact_planned_journey_;
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
            get_group_entry(pg, trp->trip_idx_);
        key.entry_station_ = entry_station;
        key.entry_time_ = entry_time;
        break;
      }
      default: break;
    }
    return key;
  };

  auto const group_enters_here =
      [&](event_node const* trp_node,
          passenger_group const* pg) -> std::pair<bool, trip const*> {
    for (auto const& [leg_idx, leg] :
         utl::enumerate(pg->compact_planned_journey_.legs_)) {
      if (leg.trip_idx_ == trp->trip_idx_ &&
          leg.enter_station_id_ == trp_node->station_idx()) {
        return {
            true,
            leg_idx > 0
                ? get_trip(
                      sched,
                      pg->compact_planned_journey_.legs_[leg_idx - 1].trip_idx_)
                : nullptr};
      }
    }
    return {false, nullptr};
  };

  auto const group_exits_here =
      [&](event_node const* trp_node,
          passenger_group const* pg) -> std::pair<bool, trip const*> {
    for (auto const& [leg_idx, leg] :
         utl::enumerate(pg->compact_planned_journey_.legs_)) {
      if (leg.trip_idx_ == trp->trip_idx_ &&
          leg.exit_station_id_ == trp_node->station_idx()) {
        return {
            true,
            leg_idx < pg->compact_planned_journey_.legs_.size() - 1
                ? get_trip(
                      sched,
                      pg->compact_planned_journey_.legs_[leg_idx + 1].trip_idx_)
                : nullptr};
      }
    }
    return {false, nullptr};
  };

  message_creator mc;

  auto const make_section_info = [&](edge const* e) {
    auto const* from = e->from(uv);
    auto const* to = e->to(uv);

    struct grouped_pgs_t {
      std::vector<passenger_group_index> groups_{};
      std::uint32_t min_pax_{};
      std::uint32_t max_pax_{};
      float avg_pax_{};
    };

    mcd::hash_map<grouped_key, grouped_pgs_t> grouped;
    std::vector<flatbuffers::Offset<GroupedPassengerGroups>> grouped_pgs_vec;

    for (auto const pgi : uv.pax_connection_info_.groups_[e->pci_]) {
      auto const* pg = uv.passenger_groups_.at(pgi);
      if (pg->probability_ == 0.0F) {
        continue;
      }
      trip const* other_trp = nullptr;

      if (grp_filter == PaxMonGroupFilter_Entering) {
        auto const [entering, ot] = group_enters_here(from, pg);
        if (entering) {
          other_trp = ot;
        } else {
          continue;
        }
      } else if (grp_filter == PaxMonGroupFilter_Exiting) {
        auto const [exiting, ot] = group_exits_here(to, pg);
        if (exiting) {
          other_trp = ot;
        } else {
          continue;
        }
      }

      if (grp_by_other_trip &&
          grp_by_station == PaxMonGroupByStation_EntryAndLast) {
        other_trp = get_trip_before_entry(sched, pg, trp->trip_idx_);
      }

      auto const key = get_key(pg, other_trp);
      auto& gg = grouped[key];
      gg.groups_.emplace_back(pgi);
      gg.max_pax_ += pg->passengers_;
      if (pg->probability_ == 1.0F) {
        gg.min_pax_ += pg->passengers_;
      }
      gg.avg_pax_ += pg->passengers_ * pg->probability_;
    }

    auto sorted_keys =
        utl::to_vec(grouped, [](auto const& kv) { return kv.first; });
    std::sort(begin(sorted_keys), end(sorted_keys),
              [&](auto const& a, auto const& b) {
                return grouped[a].max_pax_ > grouped[b].max_pax_;
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

      auto const pdf = get_load_pdf(uv.passenger_groups_, gbd.groups_);
      auto const cdf = get_cdf(pdf);

      if (include_group_infos) {
        std::sort(begin(gbd.groups_), end(gbd.groups_));
      }

      grouped_pgs_vec.emplace_back(CreateGroupedPassengerGroups(
          mc, grouped_by_station, grouped_by_trip, entry_station, entry_time,
          CreatePaxMonCombinedGroups(
              mc,
              mc.CreateVectorOfStructs(
                  include_group_infos
                      ? utl::to_vec(gbd.groups_,
                                    [&](passenger_group_index const pgi) {
                                      return to_fbs_base_info(
                                          mc, *uv.passenger_groups_[pgi]);
                                    })
                      : std::vector<PaxMonGroupBaseInfo>{}),
              to_fbs_distribution(mc, pdf, cdf))));
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
