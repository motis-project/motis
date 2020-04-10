#include "motis/loader/gtfs/common.h"

namespace motis::loader::gtfs {
std::string parse_stop_id(std::string const& stop_id) {
  auto colon_idx = stop_id.find_first_of(':');
  return colon_idx != std::string::npos ? stop_id.substr(0, colon_idx)
                                        : stop_id;
  //  return stop_id;
}
}  // namespace motis::loader::gtfs