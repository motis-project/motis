#include "motis/loader/gtfs/stop.h"

#include <algorithm>
#include <string>
#include <tuple>

#include "utl/parser/csv.h"

#include "motis/core/common/logging.h"
#include "motis/loader/gtfs/common.h"
#include "motis/loader/util.h"
#include "motis/hash_map.h"
#include "motis/string.h"

using namespace utl;
using namespace flatbuffers64;
using std::get;

namespace motis::loader::gtfs {

enum { stop_id, stop_name, stop_timezone, stop_lat, stop_lon };
using gtfs_stop = std::tuple<cstr, cstr, cstr, float, float>;
static const column_mapping<gtfs_stop> columns = {
    {"stop_id", "stop_name", "stop_timezone", "stop_lat", "stop_lon"}};

stop_map read_stops(loaded_file file, bool const shorten_stop_ids) {
  motis::logging::scoped_timer timer{"read stops"};

  stop_map stops;
  mcd::hash_map<std::string, std::vector<stop*>> equal_names;
  for (auto const& s : read<gtfs_stop>(file.content(), columns)) {
    // load the swiss dataset without track information
    auto id = parse_stop_id(shorten_stop_ids, get<stop_id>(s).to_str());
    if (auto const it = stops.find(id); it == end(stops)) {
      auto const new_stop =
          stops
              .emplace(
                  id, std::make_unique<stop>(id, get<stop_name>(s).to_str(),
                                             get<stop_lat>(s), get<stop_lon>(s),
                                             get<stop_timezone>(s).to_str()))
              .first->second.get();
      equal_names[get<stop_name>(s).view()].emplace_back(new_stop);
    }
  }

  for (auto const& [id, s] : stops) {
    for (auto const& equal : equal_names[s->name_]) {
      if (equal != s.get()) {
        s->same_name_.emplace(equal);
      }
    }
  }

  return stops;
}

}  // namespace motis::loader::gtfs
