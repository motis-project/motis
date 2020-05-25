#pragma once

#include <string>

namespace motis::loader::gtfs {

std::string parse_stop_id(bool shorten_stop_ids, std::string const&);

}  // namespace motis::loader::gtfs