#include "motis/csa/csa.h"

#include "motis/core/access/time_access.h"
#include "motis/core/journey/journeys_to_message.h"
#include "motis/module/event_collector.h"

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

void csa::import(motis::module::import_dispatcher& reg) {
  std::make_shared<event_collector>(
      get_data_directory().generic_string(), "csa", reg,
      [this](std::map<std::string, msg_ptr> const& /*dependencies*/,
             event_collector::publish_fn_t const& /*publish*/) {
        timetable_ =
            build_csa_timetable(get_sched(), bridge_zero_duration_connections_,
                                add_footpath_connections_);
        import_successful_ = true;
      })
      ->require("SCHEDULE", [](msg_ptr const& msg) {
        return msg->get()->content_type() == MsgContent_ScheduleEvent;
      });
}

void csa::init(motis::module::registry& reg) {
  reg.register_op("/csa",
                  [&](msg_ptr const& msg) {
#ifdef MOTIS_AVX
                    return route(msg, implementation_type::CPU_SSE);
#else
                    return route(msg, implementation_type::CPU);
#endif
                  },
                  {kScheduleReadAccess});
  reg.register_op(
      "/csa/cpu",
      [&](msg_ptr const& msg) { return route(msg, implementation_type::CPU); },
      {kScheduleReadAccess});

#ifdef MOTIS_AVX
  reg.register_op("/csa/cpu/sse",
                  [&](msg_ptr const& msg) {
                    return route(msg, implementation_type::CPU_SSE);
                  },
                  {kScheduleReadAccess});
#endif

#ifdef MOTIS_CUDA
  reg.register_op(
      "/csa/gpu",
      [&](msg_ptr const& msg) { return route(msg, implementation_type::GPU); },
      {kScheduleReadAccess});
#endif

  reg.register_op(
      "/csa/update_timetable",
      [&](msg_ptr const&) -> msg_ptr {
        timetable_ =
            build_csa_timetable(get_sched(), bridge_zero_duration_connections_,
                                add_footpath_connections_);
        return {};
      },
      ctx::accesses_t{ctx::access_request{
          to_res_id(::motis::module::global_res_id::SCHEDULE),
          ctx::access_t::WRITE}});
}

csa_timetable const* csa::get_timetable() const { return timetable_.get(); }

motis::module::msg_ptr csa::route(motis::module::msg_ptr const& msg,
                                  implementation_type impl_type) const {
  auto const req = motis_content(RoutingRequest, msg);
  auto const& sched = get_sched();
  auto const response = run_csa_search(
      sched, *timetable_, csa_query(sched, req), req->search_type(), impl_type);
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
