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
  raptor_query() = delete;
  raptor_query(raptor_query const&) = delete;
  raptor_query(raptor_query const&&) = delete;
  raptor_query operator=(raptor_query const&) = delete;
  raptor_query operator=(raptor_query const&&) = delete;

  raptor_query(base_query const& bq, raptor_schedule const& raptor_sched,
               raptor_timetable const& tt, bool const use_start_footpaths)
      : tt_(tt) {
    static_cast<base_query&>(*this) = bq;

    result_ = std::make_unique<raptor_result>(tt_.stop_count());

    add_starts_ = get_add_starts(raptor_sched, source_, use_start_footpaths,
                                 use_start_metas_);
  }

  ~raptor_query() = default;

  raptor_timetable const& tt_;
  std::vector<additional_start> add_starts_;
  std::unique_ptr<raptor_result> result_;
};

#if defined(MOTIS_CUDA)

struct d_query : base_query {
  d_query() = delete;

  d_query(base_query const& bq, raptor_schedule const&,
          raptor_timetable const& tt, bool const, device* device) {
    static_cast<base_query&>(*this) = bq;

    device_ = device;

    stop_count_ = tt.stop_count();

    // +1 due to scratchpad memory for GPU
    auto const arrival_bytes =
        stop_count_ * sizeof(time) * (max_raptor_round + 1);

    cuda_malloc_set(&(d_arrivals_.front()), arrival_bytes, 0xFFu);
    for (auto k = 1u; k < d_arrivals_.size(); ++k) {
      d_arrivals_[k] = d_arrivals_[k - 1] + stop_count_;
    }

    footpaths_scratchpad_ =
        d_arrivals_.front() + (d_arrivals_.size() * stop_count_);

    size_t station_byte_count = ((tt.stop_count() / 32) + 1) * 4;
    size_t route_byte_count = ((tt.route_count() / 32) + 1) * 4;

    cuda_malloc_set(&station_marks_, station_byte_count, 0);
    cuda_malloc_set(&route_marks_, route_byte_count, 0);

    cudaMalloc(&any_station_marked_d_, sizeof(bool));
    cudaMemset(any_station_marked_d_, 0, sizeof(bool));

    cudaMallocHost(&any_station_marked_h_, sizeof(bool));
    *any_station_marked_h_ = false;

    cudaStreamCreate(&proc_stream_);
    cc();

    cudaStreamCreate(&transfer_stream_);
    cc();

    result_ = new raptor_result_pinned(stop_count_);
  }

#if !defined(__CUDACC__)
  // Do not copy queries, else double free
  d_query(d_query const&) = delete;
#else
  // CUDA needs the copy constructor for the kernel call,
  // as we pass the query to the kernel, which must be a copy
  d_query(d_query const&) = default;
#endif

  __host__ __device__ ~d_query() {
// Only call free when destructor is called by host,
// not when device kernel exits, as we pass the query to the kernel
#if !defined(__CUDACC__)
    cuda_free(d_arrivals_.front());
    cuda_free(station_marks_);
    cuda_free(route_marks_);
    delete result_;
#endif
  }

  stop_id stop_count_;
  cudaStream_t proc_stream_;
  cudaStream_t transfer_stream_;

  // Pointers to device memory
  bool* any_station_marked_d_;
  arrival_ptrs d_arrivals_;
  time* footpaths_scratchpad_;
  unsigned int* station_marks_;
  unsigned int* route_marks_;

  // Pointers to host memory
  device* device_;
  raptor_result_pinned* result_;
  bool* any_station_marked_h_;
};

#endif

}  // namespace motis::raptor