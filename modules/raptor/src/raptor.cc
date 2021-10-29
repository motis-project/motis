#include "motis/raptor/raptor.h"

#include "utl/to_vec.h"

#include "motis/module/message.h"

#include "motis/core/common/timing.h"
#include "motis/core/schedule/schedule.h"
#include "motis/core/access/time_access.h"
#include "motis/core/journey/journey.h"
#include "motis/core/journey/journey_util.h"
#include "motis/core/journey/journeys_to_message.h"
#include "motis/core/journey/message_to_journeys.h"

#include "motis/raptor/get_raptor_schedule.h"
#include "motis/raptor/raptor_query.h"
#include "motis/raptor/raptor_search.h"

using namespace motis::module;
using namespace motis::routing;

namespace motis::raptor {

raptor::raptor() : module("RAPTOR Options", "raptor") {
#if defined(MOTIS_CUDA)
  param(mp_per_query_, "mp_per_query",
        "specifies how many multiprocessors are allocated to a single query");
#endif
}

raptor::~raptor() {
#if defined(MOTIS_CUDA)
  if (d_gtt_ != nullptr) {
    destroy_device_gpu_timetable(*d_gtt_);
  }
#endif
}

void raptor::init(motis::module::registry& reg) {
  auto const& sched = get_sched();

  std::tie(raptor_sched_, timetable_) = get_raptor_schedule(sched);

  reg.register_op("/raptor", [&](motis::module::msg_ptr const& msg) {
#if defined(MOTIS_CUDA)
    return route_generic<d_query>(msg, hybrid_raptor);
#else
    return route_generic<raptor_query>(msg, cpu_raptor);
#endif
  });

  reg.register_op("/raptor_cpu", [&](motis::module::msg_ptr const& msg) {
    return route_generic<raptor_query>(msg, cpu_raptor);
  });

#if defined(MOTIS_CUDA)
  h_gtt_ = get_host_gpu_timetable(*raptor_sched_, *timetable_);
  d_gtt_ = get_device_gpu_timetable(*h_gtt_);

  reg.register_op("/raptor_gpu", [&](motis::module::msg_ptr const& msg) {
    return route_generic<d_query>(msg, gpu_raptor);
  });

  reg.register_op("/raptor_hy", [&](motis::module::msg_ptr const& msg) {
    return route_generic<d_query>(msg, hybrid_raptor);
  });

  devices_ = get_devices();
  mp_per_query_ = std::max(mp_per_query_, int32_t{1});
#endif
}

template <class Query>
inline Query raptor::get_query(
    motis::routing::RoutingRequest const* routing_request,
    schedule const& sched) {
  auto base_query = get_base_query(routing_request, sched, *raptor_sched_);

  auto const use_start_footpaths = routing_request->use_start_footpaths();

  if constexpr (std::is_same_v<Query, raptor_query>) {
    return raptor_query(base_query, *raptor_sched_, *timetable_,
                        use_start_footpaths);
  } else {
    return d_query(base_query, *raptor_sched_, *timetable_, use_start_footpaths,
                   &devices_.front(), mp_per_query_);
  }
}

template <typename Query, typename RaptorFun>
msg_ptr raptor::route_generic(msg_ptr const& msg,
                              RaptorFun const& raptor_search) {
  auto const req = motis_content(RoutingRequest, msg);
  auto const& sched = get_sched();

  raptor_statistics stats;

  auto q = get_query<Query>(req, sched);
  q.id_ = (int)msg->id();

  MOTIS_START_TIMING(total_calculation_time);
  auto const& js = raptor_search(q, stats, sched, *raptor_sched_, *timetable_);
  stats.total_calculation_time_ = MOTIS_GET_TIMING_MS(total_calculation_time);

  return make_response(sched, js, req, stats);
}

msg_ptr make_response(schedule const& sched, std::vector<journey> const& js,
                      motis::routing::RoutingRequest const* request,
                      raptor_statistics const& stats) {
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