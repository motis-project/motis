#pragma once

#include <cstdint>
#include <map>
#include <tuple>
#include <vector>

#include "motis/core/schedule/connection.h"
#include "motis/core/schedule/footpath.h"
#include "motis/core/schedule/time.h"

#ifdef MOTIS_CUDA
#include "motis/csa/gpu/gpu_timetable.h"
#endif

namespace motis {

struct light_connection;
struct station;

namespace csa {

using station_id = uint32_t;
using trip_id = uint32_t;
using con_idx_t = uint16_t;

struct csa_connection {
  csa_connection() = delete;
  csa_connection(uint32_t from_station, uint32_t to_station,
                 motis::time departure, motis::time arrival, uint16_t price,
                 uint32_t trip, con_idx_t trip_con_idx, bool from_in_allowed,
                 bool to_out_allowed, service_class clasz,
                 light_connection const* light_con)
      : from_station_(from_station),
        to_station_(to_station),
        trip_(trip),
        departure_(departure),
        arrival_(arrival),
        price_(price),
        trip_con_idx_(trip_con_idx),
        from_in_allowed_(from_in_allowed),
        to_out_allowed_(to_out_allowed),
        clasz_(clasz),
        light_con_(light_con) {}
  explicit csa_connection(motis::time t) : departure_(t), arrival_(t) {}

  inline duration get_duration() const { return arrival_ - departure_; }

  station_id from_station_{0};
  station_id to_station_{0};
  trip_id trip_{0};
  motis::time departure_;
  motis::time arrival_{0};
  uint16_t price_{0};
  con_idx_t trip_con_idx_{0};
  bool from_in_allowed_{false};
  bool to_out_allowed_{false};
  service_class clasz_{service_class::OTHER};
  light_connection const* light_con_{nullptr};
};

struct csa_station {
  csa_station() = delete;
  explicit csa_station(station const* station_ptr);

  unsigned id_;
  motis::time transfer_time_;
  std::vector<footpath> footpaths_;
  std::vector<footpath> incoming_footpaths_;
  std::vector<csa_connection const*> outgoing_connections_;
  std::vector<csa_connection const*> incoming_connections_;
  station const* station_ptr_;
};

struct csa_timetable {
  std::vector<csa_station> stations_;
  std::vector<csa_connection> fwd_connections_, bwd_connections_;
  std::vector<uint32_t> fwd_bucket_starts_, bwd_bucket_starts_;

  std::vector<std::vector<csa_connection const*>> trip_to_connections_;

#ifdef MOTIS_CUDA
  gpu_timetable gpu_timetable_;
#endif

  uint32_t trip_count_{0};
};

}  // namespace csa
}  // namespace motis
