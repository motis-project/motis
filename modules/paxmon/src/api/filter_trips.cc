#include "motis/paxmon/api/filter_trips.h"

#include "utl/to_vec.h"
#include "utl/verify.h"

#include "motis/hash_set.h"

#include "motis/paxmon/get_load.h"
#include "motis/paxmon/get_universe.h"
#include "motis/paxmon/messages.h"

using namespace motis::module;
using namespace motis::paxmon;

namespace motis::paxmon::api {

msg_ptr filter_trips(schedule const& sched, paxmon_data& data,
                     msg_ptr const& msg) {
  auto const req = motis_content(PaxMonFilterTripsRequest, msg);
  auto& uv = get_universe(data, req->universe());
  auto const current_time =
      unix_to_motistime(sched.schedule_begin_, sched.system_time_);
  utl::verify(current_time != INVALID_TIME, "invalid current system time");

  auto const load_factor_threshold = req->load_factor_possibly_ge();
  auto const ignore_past_sections = req->ignore_past_sections();

  auto critical_sections = 0ULL;
  mcd::hash_set<trip const*> selected_trips;

  for (auto const& [trp, tdi] : uv.trip_data_.mapping_) {
    for (auto const& ei : uv.trip_data_.edges(tdi)) {
      auto const* e = ei.get(uv);
      if (!e->is_trip() || !e->has_capacity()) {
        continue;
      }
      if (ignore_past_sections && e->to(uv)->current_time() < current_time) {
        continue;
      }
      auto const pdf = get_load_pdf(uv.passenger_groups_,
                                    uv.pax_connection_info_.groups_[e->pci_]);
      auto const cdf = get_cdf(pdf);
      if (load_factor_possibly_ge(cdf, e->capacity(), load_factor_threshold)) {
        selected_trips.insert(trp);
        ++critical_sections;
      }
    }
  }

  message_creator mc;
  auto const selected_tsis = utl::to_vec(selected_trips, [&](trip const* trp) {
    return to_fbs_trip_service_info(mc, sched, trp);
  });

  mc.create_and_finish(MsgContent_PaxMonFilterTripsResponse,
                       CreatePaxMonFilterTripsResponse(
                           mc, selected_trips.size(), critical_sections,
                           mc.CreateVector(selected_tsis))
                           .Union());
  return make_msg(mc);
}

}  // namespace motis::paxmon::api
