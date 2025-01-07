#include "motis/odm/prima_state.h"

#include "boost/json.hpp"

#include "utl/zip.h"

namespace motis::odm {

void prima_state::blacklist_update(std::string_view json) {

  auto const update_pt_rides = [](auto& rides, auto& prev_rides,
                                  auto const& update) {
    std::swap(rides, prev_rides);
    rides.clear();
    auto prev_it = std::begin(prev_rides);
    for (auto const& stop : update) {
      for (auto const& time : stop.as_array()) {
        if (time.as_bool()) {
          rides.emplace_back(*prev_it);
        }
        ++prev_it;
        if (prev_it == end(prev_rides)) {
          return;
        }
      }
    }
  };

  auto const update_direct_rides = [](auto& rides, auto& prev_rides,
                                      auto const& update) {
    std::swap(rides, prev_rides);
    rides.clear();
    for (auto const& [prev, time] : utl::zip(prev_rides, update)) {
      if (time.as_bool()) {
        rides.emplace_back(prev);
      }
    }
  };

  auto const& o = boost::json::parse(json).as_object();
  if (o.contains("startBusStops")) {
    update_pt_rides(from_rides_, prev_from_rides_,
                    o.at("startBusStops").as_array());
  }
  if (o.contains("targetBusStops")) {
    update_pt_rides(to_rides_, prev_to_rides_,
                    o.at("targetBusStops").as_array());
  }
  if (o.contains("times")) {
    update_direct_rides(direct_rides_, prev_direct_rides_,
                        o.at("times").as_array());
  }
}

void prima_state::whitelist_update(std::string_view json [[maybe_unused]]) {
  // TODO
}

}  // namespace motis::odm