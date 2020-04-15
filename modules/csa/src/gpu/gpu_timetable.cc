#include "motis/csa/gpu/gpu_timetable.h"

#ifdef MOTIS_CUDA

#include <vector>

#include "utl/pipes/for_each.h"
#include "utl/to_vec.h"

#include "motis/csa/csa_timetable.h"
#include "motis/csa/gpu/gpu_csa.h"

namespace motis::csa {

gpu_timetable::gpu_timetable(gpu_timetable&& o) noexcept : ptr_{o.ptr_} {
  o.ptr_ = nullptr;
}

gpu_timetable& gpu_timetable::operator=(gpu_timetable&& o) noexcept {
  ptr_ = o.ptr_;
  o.ptr_ = nullptr;
  return *this;
}

gpu_timetable::gpu_timetable(csa_timetable& tt) : ptr_{nullptr} {
  if (tt.fwd_connections_.empty()) {
    return;
  }

  // Elementary connections.
  auto conns = utl::to_vec(tt.fwd_connections_, [&](csa_connection const& c) {
    return gpu_csa_con{
        c.from_station_,
        c.to_station_,
        c.trip_,
        c.departure_,
        static_cast<gpu_csa_time>(c.arrival_ +
                                  tt.stations_[c.to_station_].transfer_time_),
        c.trip_con_idx_,
        c.from_in_allowed_,
        c.to_out_allowed_};
  });

  // Copy to GPU.
  ptr_ = create_csa_gpu_timetable(
      conns.data(), tt.fwd_bucket_starts_.data(),
      static_cast<uint32_t>(tt.fwd_bucket_starts_.size()),
      static_cast<uint32_t>(conns.size()),
      static_cast<uint32_t>(tt.stations_.size()),
      static_cast<uint32_t>(tt.trip_count_));
  if (ptr_ == nullptr) {
    throw std::runtime_error{"GPU timetable creation failed"};
  }
}

gpu_timetable::~gpu_timetable() {
  free_csa_gpu_timetable(ptr_);
  ptr_ = nullptr;
}

}  // namespace motis::csa

#else

namespace motis::csa {

gpu_timetable::gpu_timetable(gpu_timetable&&) noexcept {}
gpu_timetable& gpu_timetable::operator=(gpu_timetable&&) noexcept {
  return *this;
}
gpu_timetable::gpu_timetable(csa_timetable&) {}

}  // namespace motis::csa

#endif
