#include "motis/loader/gtfs/trip.h"

#include <algorithm>
#include <tuple>

#include "utl/enumerate.h"
#include "utl/parser/csv.h"

#include "motis/core/common/logging.h"
#include "motis/core/common/projection.h"
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
  motis::logging::scoped_timer timer{"read trips"};

  trip_map trips;
  auto line = 1U;
  auto const p = projection{0.05, 0.2};
  auto const entries = read<gtfs_trip>(file.content(), columns);
  for (auto const& [i, t] : utl::enumerate(entries)) {
    std::clog << '\0' << p(i / static_cast<float>(entries.size())) << '\0';
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
