#include "motis/raptor/raptor.h"

#include "utl/to_vec.h"
#include "utl/verify.h"

#include "motis/module/message.h"

#include "motis/core/common/timing.h"
#include "motis/core/schedule/schedule.h"
#include "motis/core/access/time_access.h"
#include "motis/core/journey/journey.h"
#include "motis/core/journey/journey_util.h"
#include "motis/core/journey/journeys_to_message.h"
#include "motis/core/journey/message_to_journeys.h"

#include "motis/raptor/criteria/configs.h"
#include "motis/raptor/eval/commands.h"
#include "motis/raptor/get_raptor_timetable.h"
#include "motis/raptor/implementation_type.h"
#include "motis/raptor/raptor_query.h"
#include "motis/raptor/raptor_search.h"

using namespace motis::module;
using namespace motis::routing;

namespace motis::raptor {

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

struct raptor::impl {
  impl(schedule const& sched, [[maybe_unused]] config const& config)
      : sched_{sched} {
    std::tie(meta_info_, timetable_) = get_raptor_timetable(sched);

#if defined(MOTIS_CUDA)
    h_gtt_ = get_host_gpu_timetable(*timetable_);
    d_gtt_ = get_device_gpu_timetable(*h_gtt_);

    queries_per_device_ = std::max(config.queries_per_device_, int32_t{1});
    mem_store_.init(*meta_info_, *timetable_, queries_per_device_);

    cudaDeviceSynchronize();
#endif
  }

  msg_ptr route_cpu(msg_ptr const& msg) {
    MOTIS_START_TIMING(total_calculation_time);

    auto const req = motis_content(RoutingRequest, msg);

    utl::verify_ex(req->search_dir() == routing::SearchDir_Forward,
                   access::error::not_implemented);

    auto const base_query = get_base_query(req, sched_, *meta_info_);
    auto q = raptor_query{base_query, *meta_info_, *timetable_};

    raptor_statistics stats;
    auto const journeys = search_dispatch<implementation_type::CPU>(
        q, stats, sched_, *meta_info_, *timetable_);
    stats.total_calculation_time_ = MOTIS_GET_TIMING_MS(total_calculation_time);

    return make_response(sched_, journeys, req, stats);
  }

#if defined(MOTIS_CUDA)
  msg_ptr route_gpu(msg_ptr const& msg) {
    raptor_statistics stats;
    MOTIS_START_TIMING(total_calculation_time);

    auto const req = motis_content(RoutingRequest, msg);
    utl::verify_ex(req->search_dir() == routing::SearchDir_Forward,
                   access::error::not_implemented);

    auto base_query = get_base_query(req, sched_, *meta_info_);

    loaned_mem loan(mem_store_);

    d_query q(base_query, *meta_info_, loan.mem_, *d_gtt_);

    std::vector<journey> js;
    js = search_dispatch<implementation_type::GPU>(q, stats, sched_,
                                                   *meta_info_, *timetable_);
    stats.total_calculation_time_ = MOTIS_GET_TIMING_MS(total_calculation_time);

    return make_response(sched_, js, req, stats);
  }
#endif

  schedule const& sched_;
  std::unique_ptr<raptor_meta_info> meta_info_;
  std::unique_ptr<raptor_timetable> timetable_;

#if defined(MOTIS_CUDA)
  std::unique_ptr<host_gpu_timetable> h_gtt_;
  std::unique_ptr<device_gpu_timetable> d_gtt_;

  int32_t queries_per_device_{1};

  memory_store mem_store_;
#endif
};

raptor::raptor() : module("RAPTOR Options", "raptor") {
#if defined(MOTIS_CUDA)
  param(config_.queries_per_device_, "queries_per_device",
        "specifies how many queries should run concurrently per device");
#endif
}

#if defined(MOTIS_CUDA)
raptor::~raptor() {
  if (impl_ != nullptr && impl_->d_gtt_ != nullptr) {
    destroy_device_gpu_timetable(*(impl_->d_gtt_));
  }
}
#else
raptor::~raptor() = default;
#endif

void raptor::init(motis::module::registry& reg) {
  impl_ = std::make_unique<impl>(get_sched(), config_);

  reg.register_op("/raptor_cpu", [&](auto&& m) { return impl_->route_cpu(m); });

#if defined(MOTIS_CUDA)
  reg.register_op("/raptor", [&](auto&& m) { return impl_->route_gpu(m); });
  reg.register_op("/raptor_gpu", [&](auto&& m) { return impl_->route_gpu(m); });
#else
  reg.register_op("/raptor", [&](auto&& m) { return impl_->route_cpu(m); });
#endif
}

void raptor::reg_subc(motis::module::subc_reg& r) {
  r.register_cmd("print-raptor", "prints journeys with raptor schedule details",
                 eval::print_raptor);
  r.register_cmd("validate", "validates journeys against schedule",
                 eval::validate);
  r.register_cmd("print-raptor-route", "prints a given route from raptor timetable",
                 eval::print_raptor_route);
}

}  // namespace motis::raptor
