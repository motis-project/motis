#pragma once

#include "motis/tripbased/data.h"

#include "motis/protocol/RoutingRequest_generated.h"

namespace motis::tripbased {

struct additional_edge {
  additional_edge() = default;
  additional_edge(station_id station, duration_t dur, uint16_t price,
                  uint16_t accessibility, uint8_t transfers, int mumo_id)
      : station_id_(station),
        duration_(dur),
        price_(price),
        accessibility_(accessibility),
        transfers_(transfers),
        mumo_id_(mumo_id) {}
  additional_edge(station_id station, routing::MumoEdge const* info)
      : station_id_(station),
        duration_(info->duration()),
        price_(info->price()),
        accessibility_(info->accessibility()),
        transfers_(0),
        mumo_id_(info->mumo_id()) {}

  station_id station_id_{};
  duration_t duration_{};
  uint16_t price_{};
  uint16_t accessibility_{};
  uint8_t transfers_{};
  int mumo_id_{};
};

struct trip_based_query {
  inline bool is_ontrip() const {
    return start_type_ == routing::Start_OntripStationStart;
  }
  inline bool is_pretrip() const {
    return start_type_ == routing::Start_PretripStart;
  }

  routing::Start start_type_{routing::Start_NONE};
  station_id start_station_{};
  station_id destination_station_{};
  std::vector<station_id> meta_starts_;
  std::vector<station_id> meta_destinations_;
  bool intermodal_start_{};
  bool intermodal_destination_{};
  time start_time_{INVALID_TIME};
  time interval_begin_{INVALID_TIME};
  time interval_end_{INVALID_TIME};
  bool extend_interval_earlier_{false};
  bool extend_interval_later_{false};
  unsigned min_connection_count_{0};
  bool use_start_metas_{false};
  bool use_dest_metas_{false};
  bool use_start_footpaths_{false};
  search_dir dir_{search_dir::FWD};
  std::vector<additional_edge> start_edges_;
  std::vector<additional_edge> destination_edges_;
};

}  // namespace motis::tripbased
