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
  param(queries_per_device_, "queries_per_device",
        "specifies how many queries should run concurrently per device");
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

#if defined(MOTIS_CUDA)
  reg.register_op("/raptor", [&](motis::module::msg_ptr const& msg) {
    return route_gpu<false>(msg);
  });
#else
  reg.register_op("/raptor", [&](motis::module::msg_ptr const& msg) {
    return route_cpu(msg);
  });
#endif

  reg.register_op("/raptor_cpu", [&](motis::module::msg_ptr const& msg) {
    return route_cpu(msg);
  });

#if defined(MOTIS_CUDA)
  h_gtt_ = get_host_gpu_timetable(*raptor_sched_, *timetable_);
  d_gtt_ = get_device_gpu_timetable(*h_gtt_);

  reg.register_op("/raptor_gpu", [&](motis::module::msg_ptr const& msg) {
    return route_gpu<false>(msg);
  });

  reg.register_op("/raptor_hy", [&](motis::module::msg_ptr const& msg) {
    return route_gpu<true>(msg);
  });

  queries_per_device_ = std::max(queries_per_device_, queries_per_device_);
  mem_store_.init(*timetable_, queries_per_device_);
#endif
}

msg_ptr make_response(schedule const& sched, std::vector<journey> const& js,
                      motis::routing::RoutingRequest const* request,
                      raptor_statistics const& stats) {
  int64_t interval_start{0};
  int64_t interval_end{0};

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

msg_ptr raptor::route_cpu(msg_ptr const& msg) {
  auto const req = motis_content(RoutingRequest, msg);
  auto const& sched = get_sched();

  auto base_query = get_base_query(req, sched, *raptor_sched_);
  raptor_query q(base_query, *raptor_sched_, *timetable_);

  raptor_statistics stats;
  MOTIS_START_TIMING(total_calculation_time);
  auto const& js = cpu_raptor(q, stats, sched, *raptor_sched_, *timetable_);
  stats.total_calculation_time_ = MOTIS_GET_TIMING_MS(total_calculation_time);

  return make_response(sched, js, req, stats);
}

#if defined(MOTIS_CUDA)
template <bool UseHybridRaptor>
msg_ptr raptor::route_gpu(msg_ptr const& msg) {
  raptor_statistics stats;
  MOTIS_START_TIMING(total_calculation_time);

  auto const req = motis_content(RoutingRequest, msg);
  auto const& sched = get_sched();

  auto base_query = get_base_query(req, sched, *raptor_sched_);
  base_query.id_ = msg->id();

  loaned_mem loan(mem_store_);

  d_query q(base_query, loan.mem_, *d_gtt_);

  std::vector<journey> js;
  if constexpr (UseHybridRaptor) {
    js = hybrid_raptor(q, stats, sched, *raptor_sched_, *timetable_);
  } else {
    js = gpu_raptor(q, stats, sched, *raptor_sched_, *timetable_);
  }
  stats.total_calculation_time_ = MOTIS_GET_TIMING_MS(total_calculation_time);

  return make_response(sched, js, req, stats);
}
#endif

}  // namespace motis::raptor