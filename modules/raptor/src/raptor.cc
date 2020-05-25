#include "motis/raptor/raptor.h"

#include "utl/to_vec.h"

#include "motis/module/context/get_schedule.h"
#include "motis/module/message.h"

#include "motis/core/common/timing.h"
#include "motis/core/schedule/schedule.h"
#include "motis/core/access/time_access.h"
#include "motis/core/journey/journey.h"
#include "motis/core/journey/journey_util.h"
#include "motis/core/journey/journeys_to_message.h"
#include "motis/core/journey/message_to_journeys.h"

#include "motis/raptor/get_raptor_query.h"
#include "motis/raptor/get_raptor_schedule.h"
#include "motis/raptor/raptor_query.h"
#include "motis/raptor/raptor_search.h"

using namespace motis::module;
using namespace motis::routing;

namespace motis::raptor {

raptor::raptor() : module("RAPTOR Options", "raptor") {}

raptor::~raptor() = default;

void raptor::init(motis::module::registry& reg) {
  auto const& sched = get_sched();

  std::tie(raptor_sched_, timetable_, backward_timetable_) =
      get_raptor_schedule(sched);

  reg.register_op("/raptor_cpu", [&](motis::module::msg_ptr const& msg) {
    return route_generic<raptor_query>(msg, cpu_raptor);
  });
}

template <class Query>
inline Query raptor::get_query(
    motis::routing::RoutingRequest const* routing_request,
    schedule const& sched) {
  auto base_query = get_base_query(routing_request, sched, *raptor_sched_);

  auto const& tt = base_query.forward_ ? *timetable_ : *backward_timetable_;

  auto const use_start_footpaths = routing_request->use_start_footpaths();
  auto const use_start_metas = routing_request->use_start_metas();

  return Query(base_query, *raptor_sched_, tt, use_start_footpaths,
               use_start_metas);
}

template <typename Query, typename RaptorFun>
msg_ptr raptor::route_generic(msg_ptr const& msg,
                              RaptorFun const& raptor_search) {
  auto const req = motis_content(RoutingRequest, msg);
  auto const& sched = get_schedule();

  raptor_statistics stats;

  auto q = get_query<Query>(req, sched);

  MOTIS_START_TIMING(total_calculation_time);
  auto const& js = raptor_search(q, stats, sched, *raptor_sched_, *timetable_,
                                 *backward_timetable_);
  stats.total_calculation_time_ = MOTIS_GET_TIMING_MS(total_calculation_time);

  return make_response(js, req, stats);
}

msg_ptr raptor::make_response(std::vector<journey> const& js,
                              motis::routing::RoutingRequest const* request,
                              raptor_statistics const& stats) {
  auto const& sched = get_schedule();

  int64_t interval_start;
  int64_t interval_end;

  switch (request->start_type()) {
    case Start::Start_PretripStart: {
      auto const* start = static_cast<PretripStart const*>(request->start());
      auto const interval = start->interval();
      interval_start = interval->begin();
      interval_end = interval->end();
    } break;

    case Start::Start_OntripStationStart: {
      auto const* start =
          static_cast<OntripStationStart const*>(request->start());
      interval_start = start->departure_time();
      interval_end = start->departure_time();
    } break;

    default: {
      throw std::system_error(access::error::not_implemented);
    }
  }

  message_creator fbb;
  fbb.create_and_finish(
      MsgContent_RoutingResponse,
      CreateRoutingResponse(
          fbb,
          fbb.CreateVector(std::vector<flatbuffers::Offset<Statistics>>{
              to_fbs(fbb, to_stats_category("raptor", stats))}),
          fbb.CreateVector(utl::to_vec(
              js,
              [&](journey const& j) { return motis::to_connection(fbb, j); })),
          motis_to_unixtime(sched, interval_start),
          motis_to_unixtime(sched, interval_end),
          fbb.CreateVector(
              std::vector<flatbuffers::Offset<DirectConnection>>()))
          .Union());
  return make_msg(fbb);
}

}  // namespace motis::raptor