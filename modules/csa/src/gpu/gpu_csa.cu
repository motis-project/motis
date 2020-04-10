#include "motis/csa/gpu/gpu_csa.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <limits>

extern "C" {

//==============================================================================
// CUDA UTILITIES
//------------------------------------------------------------------------------
#define XSTR(s) STR(s)
#define STR(s) #s

#define FMT_HUMAN_READABLE "%.1f%s"
#define HUMAN_READABLE(size)                                                \
  ((size) > 1024 * 1024 * 1024)                                             \
      ? (((float)(size)) / 1024 / 1024 / 1024)                              \
      : ((size) > 1024 * 1024)                                              \
            ? (((float)(size)) / 1024 / 1024)                               \
            : ((size) > 1024) ? (((float)(size)) / 1024) : ((float)(size)), \
      ((size) > 1024 * 1024 * 1024)                                         \
          ? "GB"                                                            \
          : ((size) > 1024 * 1024) ? "MB" : ((size) > 1024) ? "kb" : "b"

#define CUDA_CALL(call)                                   \
  if ((code = call) != cudaSuccess) {                     \
    printf("CUDA error: %s at " STR(call) " %s:%d\n",     \
           cudaGetErrorString(code), __FILE__, __LINE__); \
    goto fail;                                            \
  }

#define CUDA_COPY_TO_DEVICE(type, target, source, size)                        \
  CUDA_CALL(cudaMalloc(&target, size * sizeof(type)))                          \
  CUDA_CALL(                                                                   \
      cudaMemcpy(target, source, size * sizeof(type), cudaMemcpyHostToDevice)) \
  device_bytes += size * sizeof(type);

__host__ __device__ inline int divup(int a, int b) {
  return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

//==============================================================================
// TIMETABLE
//------------------------------------------------------------------------------
struct gpu_timetable {
  struct gpu_csa_con* conns_;
  uint32_t* bucket_starts_;
  uint32_t station_count_, trip_count_, bucket_count_;
};

struct gpu_timetable* create_csa_gpu_timetable(
    struct gpu_csa_con* conns, uint32_t* bucket_starts, uint32_t bucket_count,
    uint32_t conn_count, uint32_t station_count, uint32_t trip_count) {
  size_t device_bytes = 0U;

  cudaError_t code;
  gpu_timetable* tt =
      static_cast<gpu_timetable*>(malloc(sizeof(gpu_timetable)));

  tt->station_count_ = station_count;
  tt->trip_count_ = trip_count;
  tt->bucket_count_ = bucket_count;
  tt->conns_ = nullptr;

  CUDA_COPY_TO_DEVICE(uint32_t, tt->bucket_starts_, bucket_starts,
                      bucket_count);
  CUDA_COPY_TO_DEVICE(struct gpu_csa_con, tt->conns_, conns, conn_count);

  printf("Schedule size on GPU: " FMT_HUMAN_READABLE "\n",
         HUMAN_READABLE(device_bytes));

  return tt;

fail:
  if (tt != nullptr) {
    cudaFree(tt->conns_);
  }
  free(tt);
  return nullptr;
}

void free_csa_gpu_timetable(struct gpu_timetable* tt) {
  if (tt == nullptr) {
    return;
  }
  cudaFree(tt->conns_);
  tt->conns_ = nullptr;
  tt->station_count_ = 0U;
  tt->trip_count_ = 0U;
  free(tt);
}

__device__ void atomic_min_u2(uint32_t* const addr, uint32_t const val) {
  uint32_t assumed, old_value = *addr;
  do {
    assumed = old_value;
    auto const new_value = __vminu2(old_value, val);
    if (new_value == old_value) {
      return;
    }
    old_value = atomicCAS(addr, assumed, new_value);
  } while (assumed != old_value);
}

//==============================================================================
// ALGORITHM
//------------------------------------------------------------------------------
struct d_query {
  uint32_t trip_reachable_size_;
  uint32_t station_arrivals_size_;
  uint32_t start_bucket_;
  gpu_csa_time time_limit_;
  gpu_csa_start* starts_;
  gpu_csa_time* station_arrivals_;
  gpu_csa_con_idx* trip_reachable_;
  gpu_timetable* tt_;
  uint32_t num_starts_;
  uint32_t num_queries_;
};

__global__ void initialize_starts(uint32_t num_starts, gpu_csa_start* starts,
                                  gpu_csa_time* arrivals,
                                  uint32_t num_stations) {
  auto const i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_starts) {
    auto const& start = starts[i];
    arrivals[start.query_idx_ * num_stations * (GPU_CSA_MAX_TRANSFERS + 1) +
             start.station_idx_ * (GPU_CSA_MAX_TRANSFERS + 1)] =
        start.start_time_;
  }
}

__device__ void update_con(d_query& q, uint32_t const q_idx,
                           uint32_t const transfers, uint32_t const con_idx) {
  auto const& con = q.tt_->conns_[con_idx];
  auto const is_last = transfers == GPU_CSA_MAX_TRANSFERS - 1;

  // Compute trip reachability based on segment index <= con.trip_con_idx_
  // for [transfers, transfers+1]
  auto const trp_con_idx =
      uint32_t{con.trip_con_idx_} | (uint32_t{con.trip_con_idx_} << 16U);
  auto const trip_reachable = reinterpret_cast<uint32_t*>(
      q.trip_reachable_ +
      q_idx * q.tt_->trip_count_ * (GPU_CSA_MAX_TRANSFERS + 1) +
      con.trip_ * (GPU_CSA_MAX_TRANSFERS + 1) + transfers);
  auto const
      con_is_trip_reachable =  // earliest trip segment <= con trip segment
      __vcmpleu2(*trip_reachable, trp_con_idx);

  // Compute station reachability based on earliest arrival <= con.arr_
  // for [transfers, transfers+1]
  uint32_t con_is_station_reachable{0U};
  if (con.in_allowed_) {
    auto const con_dep_time =
        (uint32_t{con.dep_}) | (uint32_t{con.dep_} << 16U);
    auto const from_station_times_ptr =
        q.station_arrivals_ +
        q_idx * q.tt_->station_count_ * (GPU_CSA_MAX_TRANSFERS + 1) +
        con.from_ * (GPU_CSA_MAX_TRANSFERS + 1) + transfers;
    auto const from_station_times =
        uint32_t{from_station_times_ptr[0]} |
        (uint32_t{from_station_times_ptr[1]} << 16U);  // no unaligned read
    con_is_station_reachable =  // from arr time <= con dep time
        __vcmpleu2(from_station_times, con_dep_time);
  }

  auto const is_reachable = con_is_station_reachable | con_is_trip_reachable;
  if (is_reachable == 0U) {
    return;  // none of both reachable -> early exit
  }

  // Update trip reachability and earliest arrival time.
  // Disable update by masking with 'is_reachable'.
  // Example: => min(OLD, mask=0xFFFF (not reachable) | NEW) = OLD
  //          => min(OLD, mask=0x0000 (reachable)     | NEW) = min(OLD, NEW)
  auto const to_station_times = reinterpret_cast<uint32_t*>(
      q.station_arrivals_ +
      q_idx * q.tt_->station_count_ * (GPU_CSA_MAX_TRANSFERS + 1) +
      con.to_ * (GPU_CSA_MAX_TRANSFERS + 1) + transfers + 1);
  auto const con_arr_time =
      ((uint32_t{con.arr_}) | ((is_last ? 0xFFFF : uint32_t{con.arr_}) << 16U));
  atomic_min_u2(trip_reachable, trp_con_idx | ~is_reachable);
  if (con.out_allowed_) {
    atomic_min_u2(to_station_times, con_arr_time | ~is_reachable);
  }
}

__global__ void update_connection(d_query& q, uint32_t bucket_start,
                                  uint32_t bucket_end) {
  auto const q_idx = blockIdx.y;
  auto const transfers = static_cast<int>(threadIdx.y * 2);
  auto const con_idx = bucket_start + blockIdx.x * blockDim.x + threadIdx.x;
  if (con_idx >= bucket_end) {
    return;
  }
  update_con(q, q_idx, transfers, con_idx);
}

__global__ void set_to_u16_max(uint16_t* ptr, uint32_t size) {
  auto const i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    ptr[i] = UINT16_MAX;
  }
}

__global__ void gpu_csa_kernel(d_query& q) {
  constexpr auto const THREADS_PER_BLOCK = 1024;

  // Initialization.
  set_to_u16_max<<<divup(q.station_arrivals_size_, THREADS_PER_BLOCK),
                   THREADS_PER_BLOCK>>>(q.station_arrivals_,
                                        q.station_arrivals_size_);
  set_to_u16_max<<<divup(q.trip_reachable_size_, THREADS_PER_BLOCK),
                   THREADS_PER_BLOCK>>>(q.trip_reachable_,
                                        q.trip_reachable_size_);
  initialize_starts<<<divup(q.num_starts_, 64), 64>>>(
      q.num_starts_, q.starts_, q.station_arrivals_, q.tt_->station_count_);

  // Iterate buckets.
  for (int bucket = q.start_bucket_; bucket != q.tt_->bucket_count_ - 1;
       ++bucket) {
    auto const from_idx = q.tt_->bucket_starts_[bucket];
    auto const to_idx = q.tt_->bucket_starts_[bucket + 1];
    if (q.tt_->conns_[from_idx].dep_ > q.time_limit_) {
      break;
    }
    auto const bucket_size = to_idx - from_idx;
    auto const num_blocks = dim3(
        divup(bucket_size, THREADS_PER_BLOCK / (GPU_CSA_MAX_TRANSFERS + 1)),
        q.num_queries_);
    auto const threads_per_block =
        dim3(THREADS_PER_BLOCK / ((GPU_CSA_MAX_TRANSFERS + 1) / 2),
             (GPU_CSA_MAX_TRANSFERS + 1) / 2);
    update_connection<<<num_blocks, threads_per_block>>>(q, from_idx, to_idx);
  }
}

gpu_csa_result gpu_csa_search(struct gpu_timetable* tt,
                              struct gpu_csa_start* starts, uint32_t num_starts,
                              uint32_t num_queries, uint32_t start_bucket,
                              gpu_csa_time time_limit) {
  size_t device_bytes = 0U;
  cudaError_t code;

  auto const station_arrivals_size =
      num_queries * tt->station_count_ * (GPU_CSA_MAX_TRANSFERS + 1);
  auto const trip_reachable_size =
      num_queries * tt->trip_count_ * (GPU_CSA_MAX_TRANSFERS + 1);

  // Host only.
  gpu_csa_result r;
  r.station_arrivals_ = nullptr;
  r.trip_reachable_ = nullptr;

  // Device only.
  d_query q;
  d_query* d_query_ptr{nullptr};
  q.trip_reachable_size_ = trip_reachable_size;
  q.station_arrivals_size_ = station_arrivals_size;
  q.start_bucket_ = start_bucket;
  q.num_queries_ = num_queries;
  q.num_starts_ = num_starts;
  q.time_limit_ = time_limit;
  q.starts_ = nullptr;
  q.tt_ = nullptr;
  q.station_arrivals_ = nullptr;

  // Allocate search data.
  CUDA_CALL(cudaGetLastError())

  CUDA_CALL(cudaMallocHost(&r.station_arrivals_,
                           station_arrivals_size * sizeof(gpu_csa_time)))
  CUDA_CALL(cudaMallocHost(&r.trip_reachable_,
                           trip_reachable_size * sizeof(gpu_csa_con_idx)))
  CUDA_CALL(cudaMalloc(
      &q.station_arrivals_,
      (1  // uneven offset for aligned writes to [transfers+1, transfers+2]
       + 1  // spill for writes to MAX_TRANSFERS + 1
       + station_arrivals_size) *
          sizeof(gpu_csa_time)))
  CUDA_CALL(cudaMalloc(&q.trip_reachable_,
                       (trip_reachable_size) * sizeof(gpu_csa_con_idx)))
  CUDA_COPY_TO_DEVICE(struct gpu_csa_start, q.starts_, starts, num_starts)
  CUDA_COPY_TO_DEVICE(gpu_timetable, q.tt_, tt, 1);

  // Start computation.
  q.station_arrivals_ += 1;
  CUDA_COPY_TO_DEVICE(struct d_query, d_query_ptr, &q, 1);
  gpu_csa_kernel<<<1, 1>>>(*d_query_ptr);
  q.station_arrivals_ -= 1;
  CUDA_CALL(cudaGetLastError())
  CUDA_CALL(cudaDeviceSynchronize())

  // Copy station arrivals and trip reachable back to host.
  CUDA_CALL(cudaMemcpy(r.station_arrivals_, q.station_arrivals_,
                       station_arrivals_size * sizeof(gpu_csa_time),
                       cudaMemcpyDeviceToHost))
  CUDA_CALL(cudaMemcpy(r.trip_reachable_, q.trip_reachable_,
                       trip_reachable_size * sizeof(gpu_csa_con_idx),
                       cudaMemcpyDeviceToHost))

  // Cleanup.
  cudaFree(q.starts_);
  cudaFree(q.station_arrivals_);
  cudaFree(q.trip_reachable_);
  cudaFree(q.tt_);
  CUDA_CALL(cudaDeviceSynchronize())
  CUDA_CALL(cudaGetLastError())

  r.station_arrivals_ += 1;

  return r;

fail:
  // Cleanup on error.
  cudaFree(q.starts_);
  cudaFree(q.station_arrivals_);
  cudaFree(q.trip_reachable_);
  cudaFree(q.tt_);
  cudaFreeHost(r.station_arrivals_);
  cudaFreeHost(r.trip_reachable_);
  return {nullptr, nullptr};
}

void gpu_csa_free_result(gpu_csa_result* r) {
  r->station_arrivals_ =
      r->station_arrivals_ == nullptr ? nullptr : r->station_arrivals_ + 1;
  cudaFreeHost(r->station_arrivals_);
  cudaFreeHost(r->trip_reachable_);
  r->station_arrivals_ = nullptr;
  r->trip_reachable_ = nullptr;
}

}  // extern "C"
