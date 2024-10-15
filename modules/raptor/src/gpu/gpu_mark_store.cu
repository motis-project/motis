#include "motis/raptor/gpu/gpu_mark_store.cuh"

#include "motis/raptor/gpu/raptor_utils.cuh"

namespace motis::raptor {

__device__ void mark(uint32_t* store, unsigned int const idx) {
  unsigned int const store_idx = (idx >> 5);  // divide by 32
  unsigned int const mask = 1 << (idx % 32);
  atomicOr(&store[store_idx], mask);
}

__device__ void unmark(uint32_t* store, unsigned int const idx) {
  unsigned int const store_idx = (idx >> 5);
  unsigned int const mask = 1 << (idx % 32);
  unsigned int const inv_mask = ~mask;
  atomicAnd(&store[store_idx], inv_mask);
}

__device__ bool marked(uint32_t const* const store, unsigned int idx) {
  unsigned int const store_idx = (idx >> 5);  // divide by 32
  unsigned int const val = store[store_idx];
  unsigned int const mask = 1 << (idx % 32);
  return static_cast<bool>(val & mask);
}

__device__ void reset_store(uint32_t* store, int const store_size) {
  auto const t_id = get_global_thread_id();
  auto const stride = get_global_stride();

  for (auto idx = t_id; idx < store_size; idx += stride) {
    store[idx] = 0;
  }
}

__device__ void convert_station_to_route_marks(uint32_t* station_marks,
                                               uint32_t* route_marks,
                                               bool* any_station_marked,
                                               device_gpu_timetable const& tt) {
  auto const global_t_id = get_global_thread_id();
  auto const global_stride = get_global_stride();
  for (auto idx = global_t_id; idx < tt.stop_count_; idx += global_stride) {
    if (marked(station_marks, idx)) {
      *any_station_marked = true;
      auto const stop = tt.stops_[idx];
      for (auto sri = stop.index_to_stop_routes_;
           sri < stop.index_to_stop_routes_ + stop.route_count_; ++sri) {
        mark(route_marks, tt.stop_routes_[sri]);
      }
    }
  }
}

__device__ void mc_convert_station_to_route_marks(
    uint32_t* station_marks, uint32_t* route_marks, bool* any_station_marked,
    bool* overall_station_marked, device_gpu_timetable const& tt,
    trait_id const trait_size) {
  auto const global_t_id = get_global_thread_id();
  auto const global_stride = get_global_stride();
  auto const max_idx = tt.stop_count_ * trait_size;
  for (auto idx = global_t_id; idx < max_idx; idx += global_stride) {
    if (marked(station_marks, idx)) {
      auto const t_offset = idx % trait_size;
      auto const s_id = idx / trait_size;

      any_station_marked[t_offset] = true;
      *(overall_station_marked) = true;

      auto const stop = tt.stops_[s_id];
      for (auto sri = stop.index_to_stop_routes_;
           sri < stop.index_to_stop_routes_ + stop.route_count_; ++sri) {
        mark(route_marks, tt.stop_routes_[sri] * trait_size + t_offset);
      }
    }
  }
}

__device__ void print_store(uint32_t* store, unsigned const size,
                            trait_id const trait_size) {
  if (get_global_thread_id() == 0) {
    for (auto idx = 0; idx < size; idx += trait_size) {
      auto const s_id = idx / trait_size;

      //      if (marked(store, idx))
      //        printf("id: %i\n", idx);

      bool wrote_s_id = false;
      for (auto t_offset = 0; t_offset < trait_size; ++t_offset) {
        if (marked(store, idx + t_offset)) {
          if (!wrote_s_id) {
            wrote_s_id = true;
            printf("id: %i;\t", s_id);
          }
          printf("%i;\t", t_offset);
        }
      }
      if (wrote_s_id) printf("\n");
      //    }
    }
  }
}

}  // namespace motis::raptor