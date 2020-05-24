#include "motis/loader/gtfs/route.h"

#include <algorithm>
#include <tuple>

#include "utl/enumerate.h"
#include "utl/parser/csv.h"

#include "motis/core/common/logging.h"
#include "motis/loader/util.h"

using namespace utl;
using std::get;

namespace motis::loader::gtfs {

// Source: https://groups.google.com/d/msg/gtfs-changes/keT5rTPS7Y0/71uMz2l6ke0J
std::map<unsigned, std::string> route::s_types_ = {
    // clang-format off
  { 0, "Str"},
  { 1, "U"},
  { 2, "DPN"},
  { 3, "Bus"},
  { 4, "Ferry"},
  { 5, "Str"},
  { 6, "Gondola, Suspended cable car"},
  { 7, "Funicular"},
  { 100, "Zug" },
  { 101, "High Speed Rail" },
  { 102, "Long Distance Trains" },
  { 103, "Inter Regional Rail" },
  { 104, "Car Transport Rail" },
  { 105, "Sleeper Rail" },
  { 106, "Regional Rail" },
  { 107, "Tourist Railway" },
  { 108, "Rail Shuttle (Within Complex)" },
  { 109, "S" },
  { 110, "Replacement Rail" },
  { 111, "Special Rail" },
  { 112, "Lorry Transport Rail" },
  { 113, "All Rails" },
  { 114, "Cross-Country Rail" },
  { 115, "Vehicle Transport Rail" },
  { 116, "Rack and Pinion Railway" },
  { 117, "Additional Rail" },
  { 200, "Coach" },
  { 201, "International Coach" },
  { 202, "National Coach" },
  { 203, "Shuttle Coach" },
  { 204, "Regional Coach" },
  { 205, "Special Coach" },
  { 206, "Sightseeing Coach" },
  { 207, "Tourist Coach" },
  { 208, "Commuter Coach" },
  { 209, "All Coachs" },
  { 300, "S" },
  { 400, "U" },
  { 401, "U" },
  { 402, "U" },
  { 403, "U" },
  { 404, "U" },
  { 500, "Metro" },
  { 600, "U" },
  { 700, "Bus" },
  { 701, "Bus" },
  { 702, "Bus" },
  { 703, "Bus" },
  { 704, "Bus" },
  { 705, "Bus" },
  { 706, "Bus" },
  { 707, "Bus" },
  { 708, "Bus" },
  { 709, "Bus" },
  { 710, "Bus" },
  { 711, "Bus" },
  { 712, "Bus" },
  { 713, "Bus" },
  { 714, "Bus" },
  { 715, "Bus" },
  { 716, "Bus" },
  { 800, "Bus" },
  { 900, "Str" },
  { 901, "Str" },
  { 902, "Str" },
  { 903, "Str" },
  { 904, "Str" },
  { 905, "Str" },
  { 906, "Str" },
  { 1000, "Water Transport" },
  { 1001, "International Car Ferry" },
  { 1002, "National Car Ferry" },
  { 1003, "Regional Car Ferry" },
  { 1004, "Local Car Ferry" },
  { 1005, "International Passenger Ferry" },
  { 1006, "National Passenger Ferry" },
  { 1007, "Regional Passenger Ferry" },
  { 1008, "Local Passenger Ferry" },
  { 1009, "Post Boat" },
  { 1010, "Train Ferry" },
  { 1011, "Road-Link Ferry" },
  { 1012, "Airport-Link Ferry" },
  { 1013, "Car High-Speed Ferry" },
  { 1014, "Passenger High-Speed Ferry" },
  { 1015, "Sightseeing Boat" },
  { 1016, "School Boat" },
  { 1017, "Cable-Drawn Boat" },
  { 1018, "River Bus" },
  { 1019, "Scheduled Ferry" },
  { 1020, "Shuttle Ferry" },
  { 1021, "All Water Transports" },
  { 1100, "Air" },
  { 1101, "International Air" },
  { 1102, "Domestic Air" },
  { 1103, "Intercontinental Air" },
  { 1104, "Domestic Scheduled Air" },
  { 1105, "Shuttle Air" },
  { 1106, "Intercontinental Charter Air" },
  { 1107, "International Charter Air" },
  { 1108, "Round-Trip Charter Air" },
  { 1109, "Sightseeing Air" },
  { 1110, "Helicopter Air" },
  { 1111, "Domestic Charter Air" },
  { 1112, "Schengen-Area Air" },
  { 1113, "Airship" },
  { 1114, "All Airs" },
  { 1200, "Ferry" },
  { 1300, "Telecabin" },
  { 1301, "Telecabin" },
  { 1302, "Cable Car" },
  { 1303, "Elevator" },
  { 1304, "Chair Lift" },
  { 1305, "Drag Lift" },
  { 1306, "Small Telecabin" },
  { 1307, "All Telecabins" },
  { 1400, "Funicular" },
  { 1401, "Funicular" },
  { 1402, "All Funicular" },
  { 1500, "Taxi" },
  { 1501, "Communal Taxi" },
  { 1502, "Water Taxi" },
  { 1503, "Rail Taxi" },
  { 1504, "Bike Taxi" },
  { 1505, "Licensed Taxi" },
  { 1506, "Private Hire Vehicle" },
  { 1507, "All Taxis" },
  { 1600, "Self Drive" },
  { 1601, "Hire Car" },
  { 1602, "Hire Van" },
  { 1603, "Hire Motorbike" },
  { 1604, "Hire Cycle" },
  { 1605, "All Self-Drive Vehicles" },
  { 1700, "Car train" }
    // clang-format on
};

std::optional<std::string> route::category() const {
  if (auto const it = s_types_.find(type_); it != end(s_types_)) {
    return it->second;
  } else {
    return std::nullopt;
  }
}

using gtfs_route = std::tuple<cstr, cstr, cstr, cstr, cstr, int>;
enum {
  route_id,
  agency_id,
  route_short_name,
  route_long_name,
  route_desc,
  route_type
};
static const column_mapping<gtfs_route> columns = {
    {"route_id", "agency_id", "route_short_name", "route_long_name",
     "route_desc", "route_type"}};

route_map read_routes(loaded_file file, agency_map const& agencies) {
  motis::logging::scoped_timer timer{"read routes"};

  route_map routes;
  auto const entries = read<gtfs_route>(file.content(), columns);
  for (auto const& [i, r] : utl::enumerate(entries)) {
    auto agency_it = agencies.find(get<agency_id>(r).to_str());
    auto agency_ptr =
        agency_it == end(agencies) ? nullptr : agency_it->second.get();
    if (agency_ptr == nullptr) {
      LOG(logging::warn) << "agency \"" << get<agency_id>(r).view()
                         << "\" not found (line=" << i << ")";
    }
    routes.emplace(
        get<route_id>(r).to_str(),
        std::make_unique<route>(
            agency_ptr, get<route_id>(r).to_str(),
            get<route_short_name>(r).to_str(), get<route_long_name>(r).to_str(),
            get<route_desc>(r).to_str(), get<route_type>(r)));
  }
  return routes;
}

}  // namespace motis::loader::gtfs
