#include "motis/loader/gtfs/transfers.h"

#include <algorithm>
#include <tuple>

#include "utl/parser/csv.h"

#include "motis/core/common/logging.h"
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
  motis::logging::scoped_timer timer{"read transfers"};
  std::map<stop_pair, transfer> transfers;
  for (auto const& t : read<gtfs_transfer>(f.content(), columns)) {
    if (get<from_stop_id>(t).to_str() != get<to_stop_id>(t).to_str()) {
      transfers.insert(std::make_pair(
          std::pair{stops.at(get<from_stop_id>(t).to_str()).get(),
                    stops.at(get<to_stop_id>(t).to_str()).get()},
          transfer(get<min_transfer_time>(t) / 60, get<transfer_type>(t))));
    }
  }
  return transfers;
}

}  // namespace motis::loader::gtfs
