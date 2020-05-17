#include "motis/loader/gtfs/stop.h"

#include <algorithm>
#include <string>
#include <tuple>

#include "utl/parser/csv.h"

#include "motis/core/common/logging.h"
#include "motis/loader/gtfs/common.h"
#include "motis/loader/util.h"

using namespace utl;
using namespace flatbuffers64;
using std::get;

namespace motis::loader::gtfs {

enum { stop_id, stop_name, stop_timezone, stop_lat, stop_lon };
using gtfs_stop = std::tuple<cstr, cstr, cstr, float, float>;
static const column_mapping<gtfs_stop> columns = {
    {"stop_id", "stop_name", "stop_timezone", "stop_lat", "stop_lon"}};

stop_map read_stops(loaded_file file) {
  motis::logging::scoped_timer timer{"read stops"};

  stop_map stops;
  for (auto const& s : read<gtfs_stop>(file.content(), columns)) {
    // load the swiss dataset without track information
    auto id = parse_stop_id(get<stop_id>(s).to_str());
    if (stops.count(id) == 0) {
      stops.emplace(id, std::make_unique<stop>(
                            id, get<stop_name>(s).to_str(), get<stop_lat>(s),
                            get<stop_lon>(s), get<stop_timezone>(s).to_str()));
    }
  }
  return stops;
}

}  // namespace motis::loader::gtfs
