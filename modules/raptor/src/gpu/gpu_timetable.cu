#include "motis/raptor/gpu/gpu_timetable.cuh"

#include "utl/concat.h"

#include "motis/raptor/gpu/cuda_util.h"
#include "motis/raptor/raptor_util.h"

namespace motis::raptor {

template <typename T>
inline void copy_vector_to_device(std::vector<T> const& vec, T** ptr) {
  static_assert(std::is_trivially_copyable_v<T>);

  const auto size_in_bytes = vec_size_bytes(vec);
  cudaMalloc(ptr, size_in_bytes);
  cuda_check();
  cudaMemcpy(*ptr, vec.data(), size_in_bytes, cudaMemcpyHostToDevice);
  cuda_check();
}

std::unique_ptr<host_gpu_timetable> get_host_gpu_timetable(
    raptor_timetable const& tt) {
  auto h_gtt = std::make_unique<host_gpu_timetable>();

  // Copy the members, which are identical on CPU and GPU
  h_gtt->stops_ = tt.stops_;
  h_gtt->routes_ = tt.routes_;

  h_gtt->route_stops_ = tt.route_stops_;
  h_gtt->stop_routes_ = tt.stop_routes_;

  // Create GPU footpaths, with from and to station
  h_gtt->footpaths_.resize(tt.footpath_count());
  for (stop_id s_id = 0; s_id < tt.stop_count(); ++s_id) {
    auto const& stop = tt.stops_[s_id];
    auto const& next_stop = tt.stops_[s_id + 1];

    for (auto foot_idx = stop.index_to_transfers_;
         foot_idx < next_stop.index_to_transfers_; ++foot_idx) {
      auto const& f = tt.footpaths_[foot_idx];
      h_gtt->footpaths_[foot_idx].from_ = s_id;
      h_gtt->footpaths_[foot_idx].to_ = f.to_;
      h_gtt->footpaths_[foot_idx].duration_ = f.duration_;
    }
  }

  // Create split stop times arrays
  h_gtt->stop_arrivals_.reserve(tt.stop_times_.size());
  h_gtt->stop_departures_.reserve(tt.stop_times_.size());
  for (auto const stop_time : tt.stop_times_) {
    h_gtt->stop_arrivals_.push_back(stop_time.arrival_);
    h_gtt->stop_departures_.push_back(stop_time.departure_);
  }

  h_gtt->stop_inb_occupancy_.reserve(tt.stop_attr_.size());
  for(auto const& stop_attr : tt.stop_attr_) {
    h_gtt->stop_inb_occupancy_.push_back(stop_attr.inbound_occupancy_);
  }

  h_gtt->transfer_times_ = tt.transfer_times_;

  return h_gtt;
}

std::unique_ptr<device_gpu_timetable> get_device_gpu_timetable(
    host_gpu_timetable const& h_gtt) {
  auto d_gtt = std::make_unique<device_gpu_timetable>();

  copy_vector_to_device(h_gtt.stops_, &(d_gtt->stops_));
  copy_vector_to_device(h_gtt.routes_, &(d_gtt->routes_));

  copy_vector_to_device(h_gtt.footpaths_, &(d_gtt->footpaths_));

  copy_vector_to_device(h_gtt.stop_arrivals_, &(d_gtt->stop_arrivals_));
  copy_vector_to_device(h_gtt.stop_departures_, &(d_gtt->stop_departures_));
  copy_vector_to_device(h_gtt.stop_inb_occupancy_,
                        &(d_gtt->stop_inb_occupancy_));

  copy_vector_to_device(h_gtt.route_stops_, &(d_gtt->route_stops_));
  copy_vector_to_device(h_gtt.stop_routes_, &(d_gtt->stop_routes_));

  d_gtt->stop_count_ = h_gtt.stop_count();
  d_gtt->route_count_ = h_gtt.route_count();
  d_gtt->footpath_count_ = h_gtt.footpaths_.size();

  copy_vector_to_device(h_gtt.transfer_times_, &(d_gtt->transfer_times_));

  return d_gtt;
}

void destroy_device_gpu_timetable(device_gpu_timetable& d_gtt) {
  cudaFree(d_gtt.stops_);
  cudaFree(d_gtt.routes_);
  cudaFree(d_gtt.footpaths_);
  cudaFree(d_gtt.transfer_times_);
  cudaFree(d_gtt.stop_arrivals_);
  cudaFree(d_gtt.stop_departures_);
  cudaFree(d_gtt.stop_inb_occupancy_);
  cudaFree(d_gtt.route_stops_);
  cudaFree(d_gtt.stop_routes_);
}

}  // namespace motis::raptor
