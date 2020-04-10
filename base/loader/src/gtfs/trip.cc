#include "motis/loader/gtfs/trip.h"

#include <algorithm>
#include <tuple>

#include "utl/parser/csv.h"

#include "motis/loader/util.h"

using namespace utl;
using std::get;

namespace motis::loader::gtfs {

enum { route_id, service_id, trip_id, trip_headsign, trip_short_name };
using gtfs_trip = std::tuple<cstr, cstr, cstr, cstr, cstr>;
static const column_mapping<gtfs_trip> columns = {
    {"route_id", "service_id", "trip_id", "trip_headsign", "trip_short_name"}};

trip_map read_trips(loaded_file file, route_map const& routes,
                    services const& services) {
  trip_map trips;
  auto line = 1U;
  for (auto const& t : read<gtfs_trip>(file.content(), columns)) {
    trips.emplace(
        get<trip_id>(t).to_str(),
        std::make_unique<trip>(
            routes.at(get<route_id>(t).to_str()).get(),
            services.traffic_days_.at(get<service_id>(t).to_str()).get(),
            get<trip_headsign>(t).to_str(), get<trip_short_name>(t).to_str(),
            ++line));
  }
  return trips;
}

}  // namespace motis::loader::gtfs
