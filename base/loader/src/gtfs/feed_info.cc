#include "motis/loader/gtfs/feed_info.h"

#include <algorithm>
#include <tuple>

#include "utl/parser/csv.h"

#include "motis/loader/util.h"

using namespace utl;
using namespace flatbuffers64;
using std::get;

namespace motis::loader::gtfs {

enum { feed_publisher_name, feed_version };
using gtfs_feed_publisher = std::tuple<cstr, cstr>;
static const column_mapping<gtfs_feed_publisher> columns = {
    {"feed_publisher_name", "feed_version"}};

feed_map read_feed_publisher(loaded_file file) {
  feed_map feeds;
  for (auto const& f : read<gtfs_feed_publisher>(file.content(), columns)) {
    feeds.emplace(get<feed_publisher_name>(f).to_str(),
                  feed{get<feed_publisher_name>(f).to_str(),
                       get<feed_version>(f).to_str()});
  }
  return feeds;
}

}  // namespace motis::loader::gtfs