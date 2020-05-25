#include "motis/loader/gtfs/transfers.h"

#include <algorithm>
#include <tuple>

#include "utl/parser/csv.h"

#include "motis/core/common/logging.h"
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
                                             stop_map const& stops,
                                             bool const shorten_stop_ids) {
  motis::logging::scoped_timer timer{"read transfers"};

  std::map<stop_pair, transfer> transfers;
  for (auto const& t : read<gtfs_transfer>(f.content(), columns)) {
    auto const from =
        parse_stop_id(shorten_stop_ids, get<from_stop_id>(t).to_str());
    auto const to =
        parse_stop_id(shorten_stop_ids, get<to_stop_id>(t).to_str());
    if (from != to) {
      transfers.emplace(
          std::pair{stops.at(from).get(), stops.at(to).get()},
          transfer(get<min_transfer_time>(t) / 60, get<transfer_type>(t)));
    }
  }
  return transfers;
}

}  // namespace motis::loader::gtfs
