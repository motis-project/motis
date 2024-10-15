#include "motis/raptor/eval/commands.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <tuple>

#include "motis/core/schedule/edges.h"
#include "motis/core/access/trip_access.h"
#include "motis/core/access/trip_iterator.h"
#include "motis/core/access/trip_section.h"
#include "motis/core/journey/journey.h"
#include "motis/core/journey/message_to_journeys.h"
#include "motis/module/message.h"
#include "motis/bootstrap/dataset_settings.h"
#include "motis/bootstrap/motis_instance.h"

#include "conf/configuration.h"
#include "conf/options_parser.h"

#include "motis/protocol/RoutingResponse_generated.h"

#include "version.h"

using namespace flatbuffers;
using namespace motis;
using namespace motis::bootstrap;
using namespace motis::module;
using namespace motis::routing;
using motis::module::make_msg;

namespace motis::raptor::eval {

struct validate_options : public conf::configuration {
  validate_options() : configuration{"Validate options"}, print_details_{true} {
    param(in_path_, "in", "Input file path");
    param(print_details_, "details", "Print details");
  }

  std::string in_path_;
  bool print_details_;
};

bool contains_loop_in_range(trip const* trip, uint32_t range_begin_s_off,
                            uint32_t range_end_s_off) {
  std::set<node_id_t> known_stations{};
  auto const edges = trip->edges_;
  if (range_end_s_off > edges->size()) {
    throw std::runtime_error("Range end offset is not within trip size!");
  }

  auto has_loop = false;
  for (uint32_t idx = range_begin_s_off;
       idx < edges->size() && idx < range_end_s_off; ++idx) {

    auto const edge = edges->at(idx);

    auto const from = edge->from_->get_station()->id_;
    auto const [_, inserted] = known_stations.insert(from);

    if (!inserted) {
      has_loop = true;
      break;
    }
  }

  auto const last_edge = edges->at(range_end_s_off - 1);
  auto const to = last_edge->to_->get_station()->id_;
  auto const [_, inserted] = known_stations.insert(to);

  return has_loop || !inserted;
}

std::tuple<bool, std::string> validate(schedule const& sched, journey const& j,
                                       bool print_details) {
  auto is_valid = true;
  auto const max_expected_occ = j.max_occupancy_;
  std::stringstream str{};

  str << "Journey with TR: " << j.trips_.size()
      << ";\tMOC: " << j.max_occupancy_ << ";\t";

  // iterate over the trips in the journey and validate the occupancy values to
  //   the expectation
  motis::time prev_station_arrival = INVALID_TIME;
  for (auto const& j_trip : j.trips_) {
    std::stringstream trip_ss{};
    auto const s_trip = get_trip(sched, j_trip.extern_trip_);
    auto const lc_idx = s_trip->lcon_idx_;
    auto const edges = s_trip->edges_;

    auto const j_from_stop = j.stops_[j_trip.from_];
    auto const j_to_stop = j.stops_[j_trip.to_];

    auto found_dep = false;
    auto found_arr = false;
    auto trip_valid = true;

    auto trip_msg_used = false;
    auto trip_msg_valid = true;
    auto trip_msg_prev_station_arrival = prev_station_arrival;
    std::string trip_msg{};

    auto check_dep_time = [&](trip::route_edge const& e) -> bool {
      auto const trip_dep_time = e->m_.route_edge_.conns_[lc_idx].d_time_;
      if (prev_station_arrival > trip_dep_time) {
        trip_ss << "\nIssue on Trip: " << j_trip.debug_
                << ";\tTrip can't be caught! Previous station arrival time: "
                << prev_station_arrival << " vs. this trip departure time "
                << trip_dep_time;
        return false;
      }
      return true;
    };

    uint32_t range_begin = std::numeric_limits<uint32_t>::max();
    uint32_t range_end = std::numeric_limits<uint32_t>::max();
    uint32_t trip_range_begin = range_begin;
    uint32_t trip_range_end = range_end;
    for (int idx = 0; idx < edges->size(); ++idx) {
      auto const& edge = edges->at(idx);

      auto const& dep_station =
          sched.stations_[edge->from_->get_station()->id_];

      if (!found_dep) {
        found_dep = dep_station->eva_nr_ == j_from_stop.eva_no_;
        if (found_dep && prev_station_arrival != INVALID_TIME) {
          // newly found the departure station now check the timing
          trip_valid = check_dep_time(edge);
        }
        range_begin = idx;
        if (!found_dep) continue;
      } else if (dep_station->eva_nr_ == j_from_stop.eva_no_) {
        // this trip contains a loop and therefore has the departure station
        // twice
        found_dep = true;
        trip_ss.str(std::string{});
        auto dt_check =
            prev_station_arrival == INVALID_TIME || check_dep_time(edge);
        trip_valid = dt_check;
        found_arr = false;
      }

      auto const& arr_station = sched.stations_[edge->to_->get_station()->id_];

      if (!found_arr) {
        // --- Check Occupancy Values
        auto trip_edge_occ = edge->m_.route_edge_.conns_[lc_idx].occupancy_;
        if (trip_edge_occ > max_expected_occ) {
          trip_ss << "\nIssue on Trip: " << j_trip.debug_ << ";\tEdge from "
                  << dep_station->eva_nr_ << " to " << arr_station->eva_nr_
                  << ";\tFound Occ of " << +trip_edge_occ;
          trip_valid = false;
        }
        // --- End Check Occupancy values

        if (arr_station->eva_nr_ == j_to_stop.eva_no_) {
          found_arr = true;
          // write new arrival time for the arrival_station;
          // this will be checked on the next trip
          prev_station_arrival = edge->m_.route_edge_.conns_[lc_idx].a_time_;
          range_end = idx + 1;

          if (trip_valid) {
            // we already found an arrival stop, but it is possible we will find
            // another one
            trip_msg_valid = trip_valid;
            trip_msg_used = true;
            trip_msg_prev_station_arrival = prev_station_arrival;
            trip_msg = trip_ss.str();
            trip_range_begin = range_begin;
            trip_range_end = range_end;
          }
        }
      }
    }

    if (!found_dep || !found_arr) {
      trip_ss << "\nIssue on Trip: " << j_trip.debug_
              << ";\tDidn't find departure or arrival station for given trip!";
      trip_valid = false;
    }

    if (!trip_valid) {
      if (trip_msg_used) {
        if (!trip_msg_valid) str << trip_msg;
        trip_valid = trip_msg_valid;
        prev_station_arrival = trip_msg_prev_station_arrival;
        range_begin = trip_range_begin;
        range_end = trip_range_end;
      } else {
        str << trip_ss.str();
      }
    }

    if (!trip_valid) {
      if (contains_loop_in_range(s_trip, range_begin, range_end)) {
        str << "\nTrip " << j_trip.debug_
            << ";\tContains a loop in range of used departure and arrival "
               "stations!";
      } else {
        str << "\nNo Loop found!";
      }
    }

    trip_msg = "";
    trip_msg_used = false;
    trip_msg_valid = true;
    trip_msg_prev_station_arrival = INVALID_TIME;
    range_begin = std::numeric_limits<uint32_t>::max();
    range_end = range_begin;
    trip_range_end = range_begin;
    trip_range_begin = range_begin;

    is_valid = is_valid && trip_valid;
  }

  if (print_details && is_valid) {
    str << "Ok\n";
  } else {
    str << "\n";
  }

  return std::make_tuple(is_valid, str.str());
}

int validate(int argc, const char** argv) {
  validate_options validator_opt;
  dataset_settings dataset_opt;

  try {
    conf::options_parser parser({&dataset_opt, &validator_opt});
    parser.read_command_line_args(argc, argv, false);

    if (parser.help()) {
      std::cout << "\n\tMOTIS " << short_version() << "\n\n";
      parser.print_help(std::cout);
      return 0;
    } else if (parser.version()) {
      std::cout << "MOTIS " << long_version() << "\n";
      return 0;
    }

    parser.read_configuration_file(true);
    parser.print_used(std::cout);
  } catch (std::exception const& e) {
    std::cout << "options error: " << e.what() << "\n";
    return 1;
  }

  motis_instance instance;
  instance.import(module_settings{}, dataset_opt,
                  import_settings{{dataset_opt.dataset_}});

  auto const& sched = instance.sched();

  std::ifstream in{validator_opt.in_path_.c_str()};
  in.exceptions(std::ifstream::failbit | std::ifstream::badbit);

  auto res_cnt = 0;
  auto conn_cnt = 0;
  auto v_conn_cnt = 0;
  auto v_res_cnt = 0;

  std::vector<int> failed_ids{};

  std::string json;
  while (!in.eof() && in.peek() != EOF) {
    ++res_cnt;
    std::getline(in, json);
    auto const message = make_msg(json);

    if (message->get()->content_type() != MsgContent_RoutingResponse) {
      throw std::runtime_error(
          "Found message with content type other than RoutingResponse!");
    }

    auto const res = motis_content(RoutingResponse, message);

    std::stringstream str{};
    str << "\nQuery ID: " << message->id() << "\n";
    str << "=====================================================\n";

    auto res_completely_valid = true;
    auto const journeys = message_to_journeys(res);
    for (auto const& j : journeys) {
      ++conn_cnt;
      auto const [valid, print] =
          validate(sched, j, validator_opt.print_details_);
      if (valid) {
        ++v_conn_cnt;
        if (validator_opt.print_details_) str << print;
      } else {
        res_completely_valid = false;
        str << print;
      }
    }

    if (res_completely_valid) {
      ++v_res_cnt;
    } else {
      failed_ids.emplace_back(message->id());
    }

    if (validator_opt.print_details_ || !res_completely_valid) {
      std::cout << str.str();
    }
  }

  std::cout << "\n\nSummary:\n"
            << "\t\tValid Responses:\t" << std::setw(7) << v_res_cnt << " / "
            << res_cnt << "\n"
            << "\t\tValid Conns.:\t\t" << std::setw(7) << v_conn_cnt << " / "
            << conn_cnt << "\n\n";

  if (!failed_ids.empty()) {
    std::cout << "IDs with failures: " << +failed_ids[0];
    for (int idx = 1, size = failed_ids.size(); idx < size; ++idx) {
      std::cout << ", " << +failed_ids[idx];
    }
  }

  return 0;
}

}