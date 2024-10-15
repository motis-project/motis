#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "geo/latlng.h"

#include "motis/core/schedule/time.h"
#include "motis/core/statistics/statistics.h"
#include "motis/intermodal/ppr_profiles.h"
#include "motis/protocol/Message_generated.h"

namespace motis::intermodal {

enum class mumo_type : int { FOOT, BIKE, CAR, CAR_PARKING, GBFS };

inline int to_int(mumo_type const type) {
  return static_cast<typename std::underlying_type<mumo_type>::type>(type);
}

inline std::string to_string(mumo_type const type) {
  static char const* strs[] = {"foot", "bike", "car", "car_parking", "gbfs"};
  return strs[to_int(type)];  // NOLINT
}

struct car_parking_edge {
  int32_t parking_id_{};
  geo::latlng parking_pos_{};
  uint16_t car_duration_{};
  uint16_t foot_duration_{};
  uint16_t foot_accessibility_{};
  uint16_t total_duration_{};
  bool uses_car_{};
};

struct mumo_edge {
  mumo_edge(std::string from, std::string to, geo::latlng const& from_pos,
            geo::latlng const& to_pos, duration const d, uint16_t accessibility,
            mumo_type const type, int const id)
      : from_(std::move(from)),
        to_(std::move(to)),
        from_pos_(from_pos),
        to_pos_(to_pos),
        duration_(d),
        accessibility_(accessibility),
        type_(type),
        id_(id) {}

  std::string from_, to_;
  geo::latlng from_pos_, to_pos_;
  duration duration_;
  uint16_t accessibility_;
  mumo_type type_;
  int id_;
  std::optional<car_parking_edge> car_parking_;
};

using appender_fun = std::function<mumo_edge&(
    std::string const&, geo::latlng const&, duration const, uint16_t const,
    mumo_type const, int const)>;

using mumo_stats_appender_fun = std::function<void(stats_category&&)>;

void make_starts(IntermodalRoutingRequest const*, geo::latlng const&,
                 appender_fun const&, mumo_stats_appender_fun const&,
                 ppr_profiles const&);
void make_dests(IntermodalRoutingRequest const*, geo::latlng const&,
                appender_fun const&, mumo_stats_appender_fun const&,
                ppr_profiles const&);

void remove_intersection(std::vector<mumo_edge>& starts,
                         std::vector<mumo_edge>& destinations,
                         geo::latlng const& query_start,
                         geo::latlng const& query_destination, SearchDir);

std::vector<flatbuffers::Offset<routing::AdditionalEdgeWrapper>> write_edges(
    flatbuffers::FlatBufferBuilder& fbb,  //
    std::vector<mumo_edge> const& starts,
    std::vector<mumo_edge> const& destinations,
    std::vector<mumo_edge const*>& edge_mapping);

}  // namespace motis::intermodal
