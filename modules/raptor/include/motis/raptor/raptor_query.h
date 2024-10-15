#pragma once

#include "motis/core/schedule/schedule.h"
#include "motis/module/message.h"
#include "motis/raptor/criteria/configs.h"

#include "motis/raptor/additional_start.h"
#include "motis/raptor/cpu/mark_store.h"
#include "motis/raptor/raptor_result.h"

#if defined(MOTIS_CUDA)
#include "motis/raptor/raptor_util.h"

#include "motis/raptor/gpu/cuda_util.h"
#include "motis/raptor/gpu/gpu_timetable.cuh"
#include "motis/raptor/gpu/memory_store.h"
#endif

namespace motis::raptor {

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

  raptor_criteria_config criteria_config_{raptor_criteria_config::Default};
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
        result_{std::make_unique<raptor_result>(tt_.stop_count(),
                                                bq.criteria_config_)},
        fp_times_(std::make_unique<cpu_mark_store>(
            tt.stop_count() * (max_raptor_round - 1) *
            get_trait_size_for_criteria_config(bq.criteria_config_))) {}

  ~raptor_query() = default;

  raptor_result_base const& result() const { return *result_; }

  raptor_timetable const& tt_;
  std::vector<additional_start> add_starts_;
  std::unique_ptr<raptor_result> result_;

  std::unique_ptr<cpu_mark_store> fp_times_;
};

#if defined(MOTIS_CUDA)
struct d_query : public base_query {
  d_query() = delete;
  d_query(base_query const& bq, raptor_meta_info const& meta_info, mem* mem,
          device_gpu_timetable const tt)
      : base_query{bq},
        mem_{mem},
        tt_{tt} {

    mem_->require_active(bq.criteria_config_);

    auto const add_starts = get_add_starts(
        meta_info, source_, use_start_footpaths_, use_start_metas_);

    cudaMemcpyAsync(mem_->active_device_->additional_starts_, add_starts.data(),
                    vec_size_bytes(add_starts), cudaMemcpyHostToDevice,
                    mem_->context_.transfer_stream_);
    cuda_sync_stream(mem_->context_.transfer_stream_);

    mem_->active_device_->additional_start_count_ = add_starts.size();
  }

  raptor_result_base const& result() const {
    return *mem_->active_host_->result_;
  }

  mem* mem_;
  device_gpu_timetable tt_;
};
#endif

}  // namespace motis::raptor
