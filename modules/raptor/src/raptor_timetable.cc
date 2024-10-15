#include "motis/raptor/raptor_timetable.h"

#include <sstream>

#include "utl/get_or_create.h"

namespace motis::raptor {

void route_mapping::insert_dbg(const std::string& dbg, route_id r_id, trip_id t_id) {
  auto& route_map = utl::get_or_create(
      trip_dbg_to_route_trips_, std::string{dbg},
      []() { return std::unordered_map<route_id, std::vector<trip_id>>{}; });

  auto& trip_vec = utl::get_or_create(
      route_map, r_id, []() { return std::vector<trip_id>{}; });

  trip_vec.emplace_back(t_id);
}

std::string route_mapping::str(const std::string& dbg) const {
  auto const& route_map = trip_dbg_to_route_trips_.at(dbg);
  std::stringstream str{};
  for(auto const& [r_id, trips] : route_map) {
    str << ";\tr_id: " << +r_id;

    if (!trips.empty()) {
      str << ";\tt_ids: " << trips[0];
      for (int idx = 1, size = trips.size(); idx < size; ++idx) {
        str << ", " << trips[idx];
      }
    }
  }

  return str.str();
}

}  // namespace motis::raptor