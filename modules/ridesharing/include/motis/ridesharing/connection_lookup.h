#pragma once

#include "motis/ridesharing/connection.h"
#include "motis/ridesharing/leg_idx.h"
#include "motis/ridesharing/lift.h"
#include "motis/ridesharing/lift_key.h"
#include "motis/ridesharing/passenger.h"
#include "motis/ridesharing/routing_result.h"

#include <sstream>
#include <unordered_map>

#include "geo/latlng.h"

namespace motis::ridesharing {

using connection_map = std::unordered_multimap<unsigned, connection>;

struct connection_lookup {
  connection_map from_to_;
  connection_map to_from_;
};

struct connection_eval {
  routing_result common_;
  routing_result from_leg_detour_;
  routing_result to_leg_detour_;

  std::string to_string() const {
    std::ostringstream strs;
    strs << common_.to_string() << " ; " << from_leg_detour_.to_string()
         << " ; " << to_leg_detour_.to_string();
    std::string dist = strs.str();
    return "{" + dist + "}";
  }
};

bool lift_economic(connection_eval const&);
uint16_t calculate_price(double detour_duration, double common_duration);
bool valid_passenger_count(lift const&, uint16_t passenger_count,
                           leg_idx from_leg, leg_idx to_leg);
bool valid_edge(lift const& li, leg_idx const from_leg, leg_idx const to_leg,
                connection_eval const& ce, uint16_t passenger_count);
bool passenger_constraints(lift const& li, leg_idx from_leg, leg_idx to_leg,
                           connection_eval const& ce);

/*void setup_acceptable_stations(
    std::vector<std::vector<routing_result>> const&,
    std::vector<geo::latlng> const&);
void setup_acceptable_stations(
    std::vector<std::vector<routing_result>> const& routing_table,
    std::vector<geo::latlng> const& station_locations, leg_idx pick_up_index,
    leg_idx drop_off_index);*/
connection_lookup setup_acceptable_stations(
    lift const& li,
    std::vector<std::vector<routing_result>> const& routing_table);
connection_map setup_to_from(connection_map const& from_to);

}  // namespace motis::ridesharing