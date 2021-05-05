#include "motis/csa/csa.h"

#include "motis/core/access/time_access.h"
#include "motis/core/journey/journeys_to_message.h"
#include "motis/module/context/get_schedule.h"

#include "motis/csa/build_csa_timetable.h"
#include "motis/csa/csa_query.h"
#include "motis/csa/csa_statistics.h"
#include "motis/csa/csa_timetable.h"
#include "motis/csa/csa_to_journey.h"
#include "motis/csa/error.h"
#include "motis/csa/run_csa_search.h"

using namespace motis::module;
using namespace motis::routing;

namespace motis::csa {

csa::csa() : module("CSA", "csa") {
  param(bridge_zero_duration_connections_, "bridge",
        "Bridge zero duration connections (required for GPU CSA)");
  param(add_footpath_connections_, "expand_footpaths",
        "Add CSA connections representing connection and footpath");
}

csa::~csa() = default;

void csa::init(motis::module::registry& reg) {
  timetable_ =
      build_csa_timetable(get_sched(), bridge_zero_duration_connections_,
                          add_footpath_connections_);
  reg.register_op("/csa", [&](msg_ptr const& msg) {
#ifdef MOTIS_AVX
    return route(msg, implementation_type::CPU_SSE);
#else
    return route(msg, implementation_type::CPU);
#endif
  });
  reg.register_op("/csa/cpu", [&](msg_ptr const& msg) {
    return route(msg, implementation_type::CPU);
  });
  reg.register_op("/csa/profile/cpu", [&](msg_ptr const& msg) {
    return route(msg, implementation_type::CPU, true);
  });

#ifdef MOTIS_AVX
  reg.register_op("/csa/cpu/sse", [&](msg_ptr const& msg) {
    return route(msg, implementation_type::CPU_SSE);
  });
#endif

#ifdef MOTIS_CUDA
  reg.register_op("/csa/gpu", [&](msg_ptr const& msg) {
    return route(msg, implementation_type::GPU);
  });
#endif
}

csa_timetable const* csa::get_timetable() const { return timetable_.get(); }

motis::module::msg_ptr csa::route(motis::module::msg_ptr const& msg,
                                  implementation_type impl_type,
                                  bool use_profile_search) const {
  auto const req = motis_content(RoutingRequest, msg);
  auto const& sched = get_schedule();
  auto const response =
      run_csa_search(sched, *timetable_, csa_query(sched, req),
                     req->search_type(), impl_type, use_profile_search);
  message_creator mc;
  mc.create_and_finish(
      MsgContent_RoutingResponse,
      CreateRoutingResponse(
          mc,
          mc.CreateVector(std::vector<flatbuffers::Offset<Statistics>>{
              to_fbs(mc, to_stats_category("csa", response.stats_))}),
          mc.CreateVector(utl::to_vec(response.journeys_,
                                      [&](auto const& cj) {
                                        return to_connection(
                                            mc, csa_to_journey(sched, cj));
                                      })),
          motis_to_unixtime(sched, response.searched_interval_.begin_),
          motis_to_unixtime(sched, response.searched_interval_.end_),
          mc.CreateVector(std::vector<flatbuffers::Offset<DirectConnection>>()))
          .Union());
  return make_msg(mc);
}

}  // namespace motis::csa
