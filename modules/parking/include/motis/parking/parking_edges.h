#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "motis/core/schedule/schedule.h"
#include "motis/core/schedule/station_lookup.h"
#include "motis/core/schedule/time.h"
#include "motis/module/message.h"
#include "motis/parking/parking_lot.h"

namespace motis::parking {

struct parking_edge_costs {
  parking_edge_costs() = default;

  parking_edge_costs(Station const* station, motis::osrm::Cost const* osrm_cost,
                     motis::ppr::Route const* ppr_route)
      : station_id_{station->id()->str()},
        station_name_{station->name()->str()},
        car_duration_{osrm_cost != nullptr ? static_cast<uint16_t>(std::round(
                                                 osrm_cost->duration() / 60))
                                           : static_cast<uint16_t>(0)},
        car_distance_{osrm_cost != nullptr ? osrm_cost->distance() : 0},
        foot_duration_{ppr_route->duration()},
        foot_accessibility_{ppr_route->accessibility()},
        foot_distance_{ppr_route->distance()} {}

  parking_edge_costs(std::string station_id, std::string station_name,
                     motis::osrm::Cost const* osrm_cost, uint16_t foot_duration,
                     uint16_t foot_accessibility, double foot_distance)
      : station_id_{std::move(station_id)},
        station_name_{std::move(station_name)},
        car_duration_{osrm_cost != nullptr ? static_cast<uint16_t>(std::round(
                                                 osrm_cost->duration() / 60))
                                           : static_cast<uint16_t>(0)},
        car_distance_{osrm_cost != nullptr ? osrm_cost->distance() : 0},
        foot_duration_{foot_duration},
        foot_accessibility_{foot_accessibility},
        foot_distance_{foot_distance} {}

  parking_edge_costs(Station const* station, uint16_t car_duration,
                     double car_distance, uint16_t foot_duration,
                     double foot_distance, uint16_t foot_accessibility)
      : station_id_{station->id()->str()},
        station_name_{station->name()->str()},
        car_duration_{car_duration},
        car_distance_{car_distance},
        foot_duration_{foot_duration},
        foot_accessibility_{foot_accessibility},
        foot_distance_{foot_distance} {}

  bool valid() const {
    return total_duration_ != std::numeric_limits<duration>::max();
  }

  std::string station_id_;
  std::string station_name_;
  uint16_t car_duration_{0};  // min
  double car_distance_{0};
  uint16_t foot_duration_{0};  // min
  uint16_t foot_accessibility_{0};
  double foot_distance_{0};
  duration total_duration_{
      std::min(static_cast<duration>(car_duration_ + foot_duration_),
               std::numeric_limits<duration>::max())};  // min
  uint16_t total_accessibility_{foot_accessibility_};
};

struct parking_edges {
  parking_edges() = default;
  parking_edges(parking_lot const& parking,
                std::vector<parking_edge_costs> outward_costs,
                std::vector<parking_edge_costs> return_costs)
      : parking_(parking),
        outward_costs_(std::move(outward_costs)),
        return_costs_(std::move(return_costs)) {}

  bool uses_car() const { return parking_.valid(); }

  parking_lot parking_;
  std::vector<parking_edge_costs> outward_costs_;
  std::vector<parking_edge_costs> return_costs_;
};

struct parking_edge_stats {
  int64_t osrm_duration_{};
  int64_t parking_edge_duration_{};
  int64_t parking_ppr_duration_{};
  int64_t nocar_ppr_duration_{};
};

struct database;

std::vector<parking_edges> get_parking_edges(
    station_lookup const&, std::vector<parking_lot> const& parkings,
    geo::latlng const& start_pos,
    flatbuffers::Vector<flatbuffers::Offset<Station>> const* dest_stations,
    int max_car_duration, motis::ppr::SearchOptions const* ppr_search_options,
    database& db, parking_edge_stats& pe_stats, bool include_outward,
    bool include_return, double walking_speed);

unsigned add_nocar_parking_edges(
    std::vector<parking_edges>& edges, geo::latlng const& start_pos,
    flatbuffers::Vector<flatbuffers::Offset<Station>> const* dest_stations,
    motis::ppr::SearchOptions const* ppr_search_options,
    parking_edge_stats& pe_stats, bool include_outward, bool include_return,
    double walking_speed);

}  // namespace motis::parking
