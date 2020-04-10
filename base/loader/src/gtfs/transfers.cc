#include "motis/loader/gtfs/transfers.h"

#include <algorithm>
#include <tuple>

#include "utl/parser/csv.h"

#include "motis/loader/gtfs/common.h"
#include "motis/loader/util.h"

using namespace utl;
using namespace flatbuffers64;
using std::get;

namespace motis::loader::gtfs {

enum { from_stop_id, to_stop_id, transfer_type, min_transfer_time };

using gtfs_transfer = std::tuple<cstr, cstr, int, int>;
static const column_mapping<gtfs_transfer> columns = {
    {"from_stop_id", "to_stop_id", "transfer_type", "min_transfer_time"}};

std::map<stop_pair, transfer> read_transfers(loaded_file f,
                                             stop_map const& stops) {
  std::map<stop_pair, transfer> transfers;
  for (auto const& t : read<gtfs_transfer>(f.content(), columns)) {
    if (parse_stop_id(get<from_stop_id>(t).to_str()) !=
        parse_stop_id(get<to_stop_id>(t).to_str())) {

      stop_pair key = std::make_pair(
          stops.at(parse_stop_id(get<from_stop_id>(t).to_str())).get(),
          stops.at(parse_stop_id(get<to_stop_id>(t).to_str())).get());
      transfers.insert(std::make_pair(
          key,
          transfer(get<min_transfer_time>(t) / 60, get<transfer_type>(t))));
    }
  }
  return transfers;
}

}  // namespace motis::loader::gtfs
