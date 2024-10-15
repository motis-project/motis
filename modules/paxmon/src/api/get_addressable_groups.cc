#include "motis/paxmon/api/get_addressable_groups.h"

#include <algorithm>
#include <set>
#include <utility>

#include "cista/reflection/comparable.h"

#include "utl/enumerate.h"
#include "utl/pipes.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "motis/hash_map.h"

#include "motis/core/schedule/time.h"
#include "motis/core/access/realtime_access.h"
#include "motis/core/access/trip_access.h"
#include "motis/core/access/trip_iterator.h"
#include "motis/core/conv/station_conv.h"
#include "motis/core/conv/trip_conv.h"

#include "motis/paxmon/get_load.h"
#include "motis/paxmon/get_universe.h"
#include "motis/paxmon/messages.h"

using namespace motis::module;
using namespace motis::paxmon;

namespace motis::paxmon::api {

namespace {

struct combined_group_info {
  std::vector<passenger_group_index> groups_{};
  pax_pdf pdf_;
  pax_cdf cdf_;
  pax_stats stats_;
};

struct feeder_key {
  CISTA_COMPARABLE()
  trip_idx_t trip_idx_{};
  std::uint32_t arrival_station_{};
  time schedule_arrival_{};
};

}  // namespace

void finish_combined_group_info(universe const& uv, combined_group_info& cgi) {
  std::sort(begin(cgi.groups_), end(cgi.groups_));
  cgi.pdf_ = get_load_pdf(uv.passenger_groups_, cgi.groups_);
  cgi.cdf_ = get_cdf(cgi.pdf_);
  cgi.stats_ = get_pax_stats(cgi.cdf_);
}

time get_current_arrival_time(schedule const& sched, trip const* trp,
                              std::uint32_t const station_id,
                              time const schedule_time) {
  for (auto const& sec : access::sections{trp}) {
    if (sec.to_station_id() == station_id) {
      auto const key = sec.ev_key_to();
      if (get_schedule_time(sched, key) == schedule_time) {
        return key.get_time();
      }
    }
  }
  throw utl::fail(
      "paxmon::api::get_addressable_groups: trip arrival event not found");
}

template <typename Key>
std::vector<Key> get_sorted_combined_group_infos(
    mcd::hash_map<Key, combined_group_info> const& cgis) {
  auto sorted =
      utl::to_vec(cgis, [](auto const& entry) { return entry.first; });
  std::sort(begin(sorted), end(sorted),
            [&](auto const& a_idx, auto const& b_idx) {
              auto const& a = cgis.at(a_idx);
              auto const& b = cgis.at(b_idx);
              return a.stats_.q95_ > b.stats_.q95_;
            });
  return sorted;
}

msg_ptr get_addressable_groups(paxmon_data& data, msg_ptr const& msg) {
  auto const req = motis_content(PaxMonGetAddressableGroupsRequest, msg);
  auto const uv_access = get_universe_and_schedule(data, req->universe());
  auto const& sched = uv_access.sched_;
  auto& uv = uv_access.uv_;
  auto const trp = from_fbs(sched, req->trip());

  message_creator mc;
  std::set<passenger_group_index> all_groups;

  auto const to_combined_group_ids_fbs = [&](combined_group_info const& cg) {
    return CreatePaxMonCombinedGroupIds(
        mc, mc.CreateVector(cg.groups_),
        to_fbs_distribution(mc, cg.pdf_, cg.stats_));
  };

  auto const make_grouped_by_feeder = [&](combined_group_info const& by_entry) {
    mcd::hash_map<feeder_key, combined_group_info> by_feeder;
    combined_group_info starting_here;

    for (auto const pgi : by_entry.groups_) {
      auto const* pg = uv.passenger_groups_[pgi];
      for (auto const& [leg_idx, leg] :
           utl::enumerate(pg->compact_planned_journey_.legs_)) {
        if (leg.trip_idx_ == trp->trip_idx_) {
          if (leg_idx == 0) {
            starting_here.groups_.emplace_back(pgi);
          } else {
            auto const& prev_leg =
                pg->compact_planned_journey_.legs_.at(leg_idx - 1);
            by_feeder[{prev_leg.trip_idx_, prev_leg.exit_station_id_,
                       prev_leg.exit_time_}]
                .groups_.emplace_back(pgi);
          }
          break;
        }
      }
    }

    for (auto& [feeder, cg] : by_feeder) {
      finish_combined_group_info(uv, cg);
    }
    finish_combined_group_info(uv, starting_here);

    auto const by_feeder_sorted = get_sorted_combined_group_infos(by_feeder);

    return std::pair{
        mc.CreateVector(utl::to_vec(
            by_feeder_sorted,
            [&](auto const& feeder) {
              auto const& cgi = by_feeder[feeder];
              auto const* feeder_trp = get_trip(sched, feeder.trip_idx_);
              return CreatePaxMonAddressableGroupsByFeeder(
                  mc, to_fbs_trip_service_info(mc, sched, feeder_trp),
                  to_fbs(mc, *sched.stations_[feeder.arrival_station_]),
                  motis_to_unixtime(sched, feeder.schedule_arrival_),
                  motis_to_unixtime(
                      sched, get_current_arrival_time(
                                 sched, feeder_trp, feeder.arrival_station_,
                                 feeder.schedule_arrival_)),
                  to_combined_group_ids_fbs(cgi));
            })),
        to_combined_group_ids_fbs(starting_here)};
  };

  auto const make_grouped_by_entry =
      [&](combined_group_info const& by_interchange) {
        mcd::hash_map<
            std::pair<std::uint32_t /* station idx */, time /* enter_time */>,
            combined_group_info>
            by_entry;

        for (auto const pgi : by_interchange.groups_) {
          auto const* pg = uv.passenger_groups_[pgi];
          for (auto const& leg : pg->compact_planned_journey_.legs_) {
            if (leg.trip_idx_ == trp->trip_idx_) {
              by_entry[{leg.enter_station_id_, leg.enter_time_}]
                  .groups_.emplace_back(pgi);
              break;
            }
          }
        }

        for (auto& [key, cg] : by_entry) {
          finish_combined_group_info(uv, cg);
        }

        auto const by_entry_sorted = get_sorted_combined_group_infos(by_entry);

        return mc.CreateVector(
            utl::to_vec(by_entry_sorted, [&](auto const& key) {
              auto const& cgi = by_entry[key];
              auto const by_feeder = make_grouped_by_feeder(cgi);
              return CreatePaxMonAddressableGroupsByEntry(
                  mc, to_fbs(mc, *sched.stations_[key.first]),
                  motis_to_unixtime(sched, key.second),
                  to_combined_group_ids_fbs(cgi), by_feeder.first,
                  by_feeder.second);
            }));
      };

  auto const make_section_info = [&](edge const* e) {
    auto const* from = e->from(uv);
    auto const* to = e->to(uv);

    mcd::hash_map<std::uint32_t /* station idx */, combined_group_info>
        by_interchange;

    for (auto const pgi : uv.pax_connection_info_.groups_[e->pci_]) {
      auto const* pg = uv.passenger_groups_.at(pgi);
      if (pg->probability_ == 0.0F) {
        continue;
      }
      auto skip = true;
      for (auto const& leg : pg->compact_planned_journey_.legs_) {
        if (skip) {
          if (leg.trip_idx_ == trp->trip_idx_) {
            skip = false;
          } else {
            continue;
          }
        }
        by_interchange[leg.exit_station_id_].groups_.emplace_back(pgi);
        all_groups.insert(pgi);
      }
    }

    for (auto& [ic_station, cg] : by_interchange) {
      finish_combined_group_info(uv, cg);
    }

    auto const by_interchange_sorted =
        get_sorted_combined_group_infos(by_interchange);

    return CreatePaxMonAddressableGroupsSection(
        mc, to_fbs(mc, from->get_station(sched)),
        to_fbs(mc, to->get_station(sched)),
        motis_to_unixtime(sched, from->schedule_time()),
        motis_to_unixtime(sched, from->current_time()),
        motis_to_unixtime(sched, to->schedule_time()),
        motis_to_unixtime(sched, to->current_time()),
        mc.CreateVector(utl::to_vec(
            by_interchange_sorted, [&](std::uint32_t const ic_station_idx) {
              auto const& cgi = by_interchange[ic_station_idx];
              return CreatePaxMonAddressableGroupsByInterchange(
                  mc, to_fbs(mc, *sched.stations_[ic_station_idx]),
                  to_combined_group_ids_fbs(cgi), make_grouped_by_entry(cgi));
            })));
  };

  auto const sections =
      utl::all(uv.trip_data_.edges(trp))  //
      | utl::transform([&](auto const e) { return e.get(uv); })  //
      | utl::remove_if([](auto const* e) { return !e->is_trip(); })  //
      | utl::transform([&](auto const* e) { return make_section_info(e); })  //
      | utl::vec();

  mc.create_and_finish(MsgContent_PaxMonGetAddressableGroupsResponse,
                       CreatePaxMonGetAddressableGroupsResponse(
                           mc, mc.CreateVector(sections),
                           mc.CreateVectorOfStructs(utl::to_vec(
                               all_groups,
                               [&](auto const pgi) {
                                 return to_fbs_base_info(
                                     mc, *uv.passenger_groups_[pgi]);
                               })))
                           .Union());
  return make_msg(mc);
}

}  // namespace motis::paxmon::api
