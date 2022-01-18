#pragma once

#include "motis/module/context/motis_call.h"
#include "motis/module/message.h"
#include "motis/module/module.h"

#include "motis/protocol/Message_generated.h"
#include "motis/ridesharing/connection_lookup.h"
#include "motis/ridesharing/lift.h"
#include "motis/ridesharing/query_response.h"
#include "motis/ridesharing/ridesharing.h"
#include "motis/ridesharing/routing_result.h"

#include <map>
#include <tuple>
#include <vector>

#include "geo/latlng.h"

using namespace motis::module;

namespace motis::ridesharing {

// using lift_it = std::_Rb_tree_const_iterator<motis::ridesharing::lift>;
using con_lookup_it = std::_Rb_tree_const_iterator<std::pair<
    const motis::ridesharing::lift_key, motis::ridesharing::connection_lookup>>;

enum query_mode { START = 0, DESTINATION = 1, BOTH = 2 };

struct query {
  query(RidesharingRequest const* const& req, int close_station_radius);

  routing_result trc(geo::latlng const& a, geo::latlng const& b) {
    double delta_lng = std::abs(a.lng_ - b.lng_);
    double delta_lat = std::abs(a.lat_ - b.lat_);
    return routing_result{100000 * delta_lng + 150000 * delta_lat,
                          1600000 * delta_lng + 2400000 * delta_lat};
  }

  ridesharing_edge make_direct_ridesharing_edge(std::string const& from,
                                                std::string const& to,
                                                lift const& li,
                                                connection_eval const& ce,
                                                double const approach_time,
                                                connection const& con);
  ridesharing_edge make_ridesharing_edge(
      message_creator& mc, std::string const& from, std::string const& to,
      lift const& li, connection_eval const& ce, double const approach_time,
      connection const& con, geo::latlng const& parking, int const parking_id,
      geo::latlng const& station_location);
  query_response apply(
      message_creator& mc,
      std::map<lift_key, connection_lookup> const& lift_connections,
      std::unique_ptr<database> const& db,
      std::vector<geo::latlng> const& station_locations,
      std::vector<std::pair<geo::latlng, int>> const& parkings,
      std::unordered_map<std::string, int> const& lookup_station_evas,
      std::vector<std::string> const& station_evas);
  auto close_stations(
      std::unordered_map<std::string, int> const& lookup_station_evas);
  void routing(std::vector<std::pair<geo::latlng, int>> const& parkings);
  void setup_potential_edges(con_lookup_it& low, con_lookup_it const& high,
                             std::unique_ptr<database> const& db,
                             std::vector<unsigned> const& start_station_ids,
                             std::vector<unsigned> const& target_station_ids);
  connection_eval evaluate_same_leg(query_mode mode,
                                    routing_result const& common,
                                    lift const& li, unsigned lift_idx,
                                    connection con, unsigned st,
                                    int direct_connection = -1);
  connection_eval evaluate_different_leg(query_mode mode,
                                         routing_result const& common,
                                         lift const& li, unsigned lift_idx,
                                         connection con, unsigned st,
                                         int direct_connection = -1,
                                         unsigned j = 0);

  query_response make_edges(
      message_creator& mc, std::vector<std::string> const& station_evas,
      std::vector<std::pair<geo::latlng, int>> const& parkings,
      std::vector<geo::latlng> const& station_locations);
  Position start_;
  Position target_;
  long t_;
  uint16_t passengers_;
  query_mode mode_;
  motis::ppr::SearchOptions const* ppr_search_options_;
  int close_station_radius_;
  uint64_t routing_time_query_{};
  uint64_t close_station_time_query_{};
  uint64_t parking_time_db_{};
  uint64_t parking_time_not_db_{};
  uint64_t parking_db_{};
  uint64_t parking_not_db_{};
  std::vector<std::tuple<lift, connection, int>> lift_depart_;
  std::vector<std::tuple<lift, connection>> lift_dest_;
  std::map<unsigned int, unsigned int> drop_off_stations_;
  std::map<unsigned int, unsigned int> pick_up_stations_;
  std::vector<routing_result> lifts_to_query_start_;
  std::vector<routing_result> to_drop_off_;
  std::vector<routing_result> from_pick_up_;
  std::vector<routing_result> query_target_to_lifts_;
};

}  // namespace motis::ridesharing