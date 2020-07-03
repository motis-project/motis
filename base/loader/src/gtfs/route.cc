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
std::map<unsigned, category> route::s_types_ = {
    {0, category{"Str", category::PRINT_ID | category::BASIC_ROUTE_TYPE}},
    {1, category{"U", category::PRINT_ID | category::BASIC_ROUTE_TYPE}},
    {2, category{"DPN", category::PRINT_ID | category::BASIC_ROUTE_TYPE}},
    {3, category{"Bus", category::output::PRINT_CATEGORY_AND_ID |
                            category::BASIC_ROUTE_TYPE}},
    {4, category{"Ferry", category::output::PRINT_CATEGORY_AND_ID |
                              category::BASIC_ROUTE_TYPE}},
    {5, category{"Str", category::output::PRINT_CATEGORY_AND_ID |
                            category::BASIC_ROUTE_TYPE}},
    {6, category{"Gondola, Suspended cable car",
                 category::PRINT_ID | category::BASIC_ROUTE_TYPE}},
    {7, category{"Funicular", category::PRINT_ID | category::BASIC_ROUTE_TYPE}},
    {100, category{"Railway Service"}},
    {101, category{/* "Long Distance Trains" hack for DELFI */ "ICE",
                   category::ROUTE_NAME_SHORT_INSTEAD_OF_CATEGORY |
                       category::FORCE_TRAIN_NUMBER_INSTEAD_OF_LINE_ID}},
    {102, category{/* "Long Distance Trains" hack for DELFI */ "IC",
                   category::ROUTE_NAME_SHORT_INSTEAD_OF_CATEGORY |
                       category::FORCE_TRAIN_NUMBER_INSTEAD_OF_LINE_ID}},
    {103, category{"Inter Regional Rail",
                   category::ROUTE_NAME_SHORT_INSTEAD_OF_CATEGORY |
                       category::FORCE_TRAIN_NUMBER_INSTEAD_OF_LINE_ID}},
    {104, category{"Car Transport Rail",
                   category::ROUTE_NAME_SHORT_INSTEAD_OF_CATEGORY |
                       category::FORCE_TRAIN_NUMBER_INSTEAD_OF_LINE_ID}},
    {105,
     category{"Sleeper Rail", category::ROUTE_NAME_SHORT_INSTEAD_OF_CATEGORY}},
    {106, category{"Regional Rail"}},
    {107, category{"Tourist Railway"}},
    {108, category{"Rail Shuttle (Within Complex)"}},
    {109, category{"S"}},
    {110, category{"Replacement Rail"}},
    {111, category{"Special Rail"}},
    {112, category{"Lorry Transport Rail"}},
    {113, category{"All Rails"}},
    {114, category{"Cross-Country Rail"}},
    {115, category{"Vehicle Transport Rail"}},
    {116, category{"Rack and Pinion Railway"}},
    {117, category{"Additional Rail"}},
    {200, category{"Coach"}},
    {201, category{"International Coach"}},
    {202, category{"National Coach"}},
    {203, category{"Shuttle Coach"}},
    {204, category{"Regional Coach"}},
    {205, category{"Special Coach"}},
    {206, category{"Sightseeing Coach"}},
    {207, category{"Tourist Coach"}},
    {208, category{"Commuter Coach"}},
    {209, category{"All Coachs"}},
    {300, category{"S"}},
    {400, category{"U"}},
    {401, category{"U"}},
    {402, category{"U"}},
    {403, category{"U"}},
    {404, category{"U"}},
    {500, category{"Metro"}},
    {600, category{"U"}},
    {700, category{"Bus", 0}},
    {701, category{"Bus", 0}},
    {702, category{"Bus", 0}},
    {703, category{"Bus", 0}},
    {704, category{"Bus", 0}},
    {705, category{"Bus", 0}},
    {706, category{"Bus", 0}},
    {707, category{"Bus", 0}},
    {708, category{"Bus", 0}},
    {709, category{"Bus", 0}},
    {710, category{"Bus", 0}},
    {711, category{"Bus", 0}},
    {712, category{"Bus", 0}},
    {713, category{"Bus", 0}},
    {714, category{"Bus", 0}},
    {715, category{"Bus", 0}},
    {716, category{"Bus", 0}},
    {800, category{"Bus", 0}},
    {900, category{"Str", 0}},
    {901, category{"Str", 0}},
    {902, category{"Str", 0}},
    {903, category{"Str", 0}},
    {904, category{"Str", 0}},
    {905, category{"Str", 0}},
    {906, category{"Str", 0}},
    {1000, category{"Water Transport"}},
    {1001, category{"International Car Ferry"}},
    {1002, category{"National Car Ferry"}},
    {1003, category{"Regional Car Ferry"}},
    {1004, category{"Local Car Ferry"}},
    {1005, category{"International Passenger Ferry"}},
    {1006, category{"National Passenger Ferry"}},
    {1007, category{"Regional Passenger Ferry"}},
    {1008, category{"Local Passenger Ferry"}},
    {1009, category{"Post Boat"}},
    {1010, category{"Train Ferry"}},
    {1011, category{"Road-Link Ferry"}},
    {1012, category{"Airport-Link Ferry"}},
    {1013, category{"Car High-Speed Ferry"}},
    {1014, category{"Passenger High-Speed Ferry"}},
    {1015, category{"Sightseeing Boat"}},
    {1016, category{"School Boat"}},
    {1017, category{"Cable-Drawn Boat"}},
    {1018, category{"River Bus"}},
    {1019, category{"Scheduled Ferry"}},
    {1020, category{"Shuttle Ferry"}},
    {1021, category{"All Water Transports"}},
    {1100, category{"Air"}},
    {1101, category{"International Air"}},
    {1102, category{"Domestic Air"}},
    {1103, category{"Intercontinental Air"}},
    {1104, category{"Domestic Scheduled Air"}},
    {1105, category{"Shuttle Air"}},
    {1106, category{"Intercontinental Charter Air"}},
    {1107, category{"International Charter Air"}},
    {1108, category{"Round-Trip Charter Air"}},
    {1109, category{"Sightseeing Air"}},
    {1110, category{"Helicopter Air"}},
    {1111, category{"Domestic Charter Air"}},
    {1112, category{"Schengen-Area Air"}},
    {1113, category{"Airship"}},
    {1114, category{"All Airs"}},
    {1200, category{"Ferry", 0}},
    {1300, category{"Telecabin"}},
    {1301, category{"Telecabin"}},
    {1302, category{"Cable Car"}},
    {1303, category{"Elevator"}},
    {1304, category{"Chair Lift"}},
    {1305, category{"Drag Lift"}},
    {1306, category{"Small Telecabin"}},
    {1307, category{"All Telecabins"}},
    {1400, category{"Funicular"}},
    {1401, category{"Funicular"}},
    {1402, category{"All Funicular"}},
    {1500, category{"Taxi"}},
    {1501, category{"Communal Taxi"}},
    {1502, category{"Water Taxi"}},
    {1503, category{"Rail Taxi"}},
    {1504, category{"Bike Taxi"}},
    {1505, category{"Licensed Taxi"}},
    {1506, category{"Private Hire Vehicle"}},
    {1507, category{"All Taxis"}},
    {1600, category{"Self Drive"}},
    {1601, category{"Hire Car"}},
    {1602, category{"Hire Van"}},
    {1603, category{"Hire Motorbike"}},
    {1604, category{"Hire Cycle"}},
    {1605, category{"All Self-Drive Vehicles"}},
    {1700, category{"Car train"}}};

std::optional<category> route::get_category() const {
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
    auto const agency_it = agencies.find(get<agency_id>(r).to_str());
    auto const agency_ptr =
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
