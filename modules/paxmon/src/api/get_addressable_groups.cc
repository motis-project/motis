#include "motis/paxmon/api/get_addressable_groups.h"

#include <algorithm>
#include <set>

#include "utl/pipes.h"
#include "utl/to_vec.h"

#include "motis/hash_map.h"

#include "motis/core/conv/station_conv.h"
#include "motis/core/conv/trip_conv.h"

#include "motis/paxmon/get_load.h"
#include "motis/paxmon/get_universe.h"
#include "motis/paxmon/messages.h"

using namespace motis::module;
using namespace motis::paxmon;

namespace motis::paxmon::api {

struct combined_group_info {
  std::vector<passenger_group_index> groups_{};
  pax_pdf pdf_;
  pax_cdf cdf_;
  pax_stats stats_;
};

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
        to_fbs_distribution(mc, cg.pdf_, cg.cdf_, cg.stats_));
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
      std::sort(begin(cg.groups_), end(cg.groups_));
      cg.pdf_ = get_load_pdf(uv.passenger_groups_, cg.groups_);
      cg.cdf_ = get_cdf(cg.pdf_);
      cg.stats_ = get_pax_stats(cg.cdf_);
    }
    auto by_interchange_sorted = utl::to_vec(
        by_interchange, [](auto const& entry) { return entry.first; });
    std::sort(begin(by_interchange_sorted), end(by_interchange_sorted),
              [&](std::uint32_t const a_idx, std::uint32_t const b_idx) {
                auto const& a = by_interchange[a_idx];
                auto const& b = by_interchange[b_idx];
                return a.stats_.q95_ > b.stats_.q95_;
              });

    return CreatePaxMonAddressableGroupsSection(
        mc, to_fbs(mc, from->get_station(sched)),
        to_fbs(mc, to->get_station(sched)),
        motis_to_unixtime(sched, from->schedule_time()),
        motis_to_unixtime(sched, from->current_time()),
        motis_to_unixtime(sched, to->schedule_time()),
        motis_to_unixtime(sched, to->current_time()),
        mc.CreateVector(utl::to_vec(
            by_interchange_sorted, [&](std::uint32_t const ic_station_idx) {
              return CreatePaxMonAddressableGroupsByInterchange(
                  mc, to_fbs(mc, *sched.stations_[ic_station_idx]),
                  to_combined_group_ids_fbs(by_interchange[ic_station_idx]),
                  mc.CreateVector(std::vector<flatbuffers::Offset<
                                      PaxMonAddressableGroupsByEntry>>{}));
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
