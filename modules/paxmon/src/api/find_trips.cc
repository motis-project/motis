#include "motis/paxmon/api/find_trips.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "motis/paxmon/get_universe.h"
#include "motis/paxmon/messages.h"

using namespace motis::module;
using namespace motis::paxmon;

namespace motis::paxmon::api {

msg_ptr find_trips(paxmon_data& data, msg_ptr const& msg) {
  auto const req = motis_content(PaxMonFindTripsRequest, msg);
  auto const uv_access = get_universe_and_schedule(data, req->universe());
  auto const& sched = uv_access.sched_;
  auto& uv = uv_access.uv_;

  message_creator mc;
  std::vector<flatbuffers::Offset<PaxMonTripInfo>> trips;
  auto const search_entry = std::make_pair(
      primary_trip_id{0U, req->train_nr(), 0U}, static_cast<trip*>(nullptr));
  for (auto it = std::lower_bound(begin(sched.trips_), end(sched.trips_),
                                  search_entry);
       it != end(sched.trips_) && it->first.train_nr_ == req->train_nr();
       ++it) {
    auto const trp = static_cast<trip const*>(it->second);
    if (trp->edges_->empty()) {
      continue;
    }
    auto const tdi = uv.trip_data_.find_index(trp->trip_idx_);
    auto const has_paxmon_data = tdi != INVALID_TRIP_DATA_INDEX;
    if (req->only_trips_with_paxmon_data() && !has_paxmon_data) {
      continue;
    }
    auto const service_infos = get_service_infos(sched, trp);
    if (req->filter_class()) {
      if (std::any_of(begin(service_infos), end(service_infos),
                      [&](auto const& p) {
                        return static_cast<service_class_t>(p.first.clasz_) >
                               static_cast<service_class_t>(req->max_class());
                      })) {
        continue;
      }
    }
    auto all_edges_have_capacity_info = false;
    auto has_passengers = false;
    if (has_paxmon_data) {
      auto const td_edges = uv.trip_data_.edges(tdi);
      all_edges_have_capacity_info =
          std::all_of(begin(td_edges), end(td_edges), [&](auto const& ei) {
            auto const* e = ei.get(uv);
            return !e->is_trip() || e->has_capacity();
          });
      has_passengers =
          std::any_of(begin(td_edges), end(td_edges), [&](auto const& ei) {
            auto const* e = ei.get(uv);
            return e->is_trip() &&
                   !uv.pax_connection_info_.group_routes(e->pci_).empty();
          });
    }

    trips.emplace_back(CreatePaxMonTripInfo(
        mc, to_fbs_trip_service_info(mc, sched, trp, service_infos),
        has_paxmon_data, all_edges_have_capacity_info, has_passengers));
  }

  mc.create_and_finish(
      MsgContent_PaxMonFindTripsResponse,
      CreatePaxMonFindTripsResponse(mc, mc.CreateVector(trips)).Union());
  return make_msg(mc);
}

}  // namespace motis::paxmon::api
