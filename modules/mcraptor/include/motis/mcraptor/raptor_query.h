#pragma once

#include "motis/core/schedule/schedule.h"
#include "motis/module/message.h"

#include "motis/mcraptor/additional_start.h"
#include "motis/mcraptor/raptor_result.h"
#include "Rounds.h"

#if defined(MOTIS_CUDA)
#include "motis/mcraptor/raptor_util.h"

#include "motis/mcraptor/gpu/cuda_util.h"
#include "motis/mcraptor/gpu/gpu_timetable.cuh"
#include "motis/mcraptor/gpu/memory_store.h"
#endif

namespace motis::mcraptor {

struct base_query {
  bool ontrip_{true};

  stop_id source_{invalid<stop_id>};
  stop_id target_{invalid<stop_id>};

  time source_time_begin_{invalid<time>};
  time source_time_end_{invalid<time>};

  bool forward_{true};

  bool use_start_metas_{true};
  bool use_dest_metas_{true};

  bool use_start_footpaths_{true};
};

base_query get_base_query(routing::RoutingRequest const* routing_request,
                          schedule const& sched,
                          raptor_meta_info const& meta_info);

struct raptor_query : public base_query {
  raptor_query() = delete;
  raptor_query(raptor_query const&) = delete;
  raptor_query(raptor_query const&&) = delete;
  raptor_query operator=(raptor_query const&) = delete;
  raptor_query operator=(raptor_query const&&) = delete;

  raptor_query(base_query const& bq, raptor_meta_info const& meta_info,
               raptor_timetable const& tt)
      : base_query{bq},
        tt_{tt},
        add_starts_{get_add_starts(meta_info, source_, use_start_footpaths_,
                                   use_start_metas_)},
        result_{std::make_unique<Rounds>(tt_.stop_count())} {}

  ~raptor_query() = default;

  Rounds& result() const { return *result_; }

  raptor_timetable const& tt_;
  std::vector<additional_start> add_starts_;
  std::unique_ptr<Rounds> result_;
};

#if defined(MOTIS_CUDA)
struct d_query : public base_query {
  d_query() = delete;
  d_query(base_query const& bq, raptor_meta_info const& meta_info, mem* mem,
          device_gpu_timetable const tt)
      : base_query{bq}, mem_{mem}, tt_{tt} {

    auto const& add_starts = get_add_starts(
        meta_info, source_, use_start_footpaths_, use_start_metas_);

    cudaMemcpyAsync(mem_->device_.additional_starts_, add_starts.data(),
                    vec_size_bytes(add_starts), cudaMemcpyHostToDevice,
                    mem_->context_.transfer_stream_);
    cuda_sync_stream(mem_->context_.transfer_stream_);

    mem_->device_.additional_start_count_ = add_starts.size();
  }

  raptor_result_base const& result() const { return *mem_->host_.result_; }

  mem* mem_;
  device_gpu_timetable tt_;
};
#endif

}  // namespace motis::mcraptor
