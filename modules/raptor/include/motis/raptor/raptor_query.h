#pragma once

#include "utl/to_vec.h"

#include "motis/core/schedule/schedule.h"
#include "motis/module/message.h"

#include "motis/raptor/mark_store.h"
#include "motis/raptor/raptor_result.h"
#include "motis/raptor/raptor_util.h"

#if defined(MOTIS_CUDA)
#include "motis/raptor/gpu/cuda_util.h"
#include "motis/raptor/gpu/devices.h"
#include "motis/raptor/gpu/gpu_timetable.cuh"
#include "motis/raptor/memory_store.h"
#endif

namespace motis::raptor {

struct base_query {
  int id_{-1};

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
                          raptor_schedule const& raptor_sched);

struct additional_start {
  additional_start() = delete;
  additional_start(stop_id const s_id, time const offset)
      : s_id_(s_id), offset_(offset) {}
  stop_id s_id_{invalid<stop_id>};
  time offset_{invalid<time>};
};

auto inline get_add_starts(raptor_schedule const& raptor_sched,
                           stop_id const source, bool const use_start_footpaths,
                           bool const use_start_metas) {
  std::vector<additional_start> add_starts;

  if (use_start_footpaths) {
    auto const& init_footpaths = raptor_sched.initialization_footpaths_[source];
    append_vector(add_starts, utl::to_vec(init_footpaths, [](auto const f) {
                    return additional_start(f.to_, f.duration_);
                  }));
  }

  if (use_start_metas) {
    auto const& equis = raptor_sched.equivalent_stations_[source];
    append_vector(add_starts, utl::to_vec(equis, [](auto const s_id) {
                    return additional_start(s_id, 0);
                  }));

    // Footpaths from meta stations
    if (use_start_footpaths) {
      for (auto const equi : equis) {
        auto const& init_footpaths =
            raptor_sched.initialization_footpaths_[equi];
        append_vector(add_starts, utl::to_vec(init_footpaths, [](auto const f) {
                        return additional_start(f.to_, f.duration_);
                      }));
      }
    }
  }

  return add_starts;
}

struct raptor_query : base_query {
  raptor_query() = default;
  raptor_query(raptor_query const&) = delete;
  raptor_query(raptor_query const&&) = delete;
  raptor_query operator=(raptor_query const&) = delete;
  raptor_query operator=(raptor_query const&&) = delete;

  raptor_query(base_query const& bq, raptor_schedule const& raptor_sched,
               raptor_timetable const& tt)
      : tt_(tt) {
    static_cast<base_query&>(*this) = bq;

    result_ = std::make_unique<raptor_result>(tt_.stop_count());

    add_starts_ = get_add_starts(raptor_sched, source_, use_start_footpaths_,
                                 use_start_metas_);
  }

  ~raptor_query() = default;

  raptor_result_base const& result() const { return *result_; }

  raptor_timetable const& tt_;
  std::vector<additional_start> add_starts_;
  std::unique_ptr<raptor_result> result_;
};

#if defined(MOTIS_CUDA)

struct d_query : base_query {
  d_query() = delete;
  d_query(base_query const& bq, mem* mem) : mem_(mem) {
    static_cast<base_query&>(*this) = bq;
  }

  raptor_result_base const& result() const { return *mem_->host_.result_; }

  mem* mem_;
};

#endif

}  // namespace motis::raptor