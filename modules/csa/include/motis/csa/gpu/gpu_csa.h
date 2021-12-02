#pragma once

#include <cinttypes>

extern "C" {

enum { GPU_CSA_MAX_TRANSFERS = 7U };
enum { GPU_CSA_MAX_TRAVEL_TIME = 1440U };

struct gpu_timetable;

using gpu_csa_time = uint16_t;
using gpu_csa_station_id = uint32_t;
using gpu_csa_trip_idx = uint32_t;
using gpu_csa_con_idx = uint16_t;

struct gpu_csa_con {
  gpu_csa_station_id from_, to_;
  gpu_csa_trip_idx trip_;
  gpu_csa_time dep_, arr_;
  gpu_csa_con_idx trip_con_idx_;
  bool in_allowed_, out_allowed_;
};

struct gpu_timetable* create_csa_gpu_timetable(
    struct gpu_csa_con* conns, uint32_t* bucket_starts, uint32_t bucket_count,
    uint32_t conn_count, uint32_t station_count, uint32_t trip_count);

void free_csa_gpu_timetable(struct gpu_timetable*);

struct gpu_csa_result {
  gpu_csa_time* station_arrivals_;
  gpu_csa_con_idx* trip_reachable_;
};

struct gpu_csa_start {
  uint32_t query_idx_;
  gpu_csa_station_id station_idx_;
  gpu_csa_time start_time_;
};

gpu_csa_result gpu_csa_search(struct gpu_timetable*, struct gpu_csa_start*,
                              uint32_t num_starts, uint32_t num_queries,
                              uint32_t start_bucket, gpu_csa_time time_limit);

void gpu_csa_free_result(gpu_csa_result*);

}  // extern "C"
