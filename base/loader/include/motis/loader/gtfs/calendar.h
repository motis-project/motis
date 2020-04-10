#pragma once

#include <bitset>
#include <map>
#include <string>

#include "boost/date_time/gregorian/gregorian_types.hpp"

#include "motis/loader/loaded_file.h"

namespace motis::loader::gtfs {

struct calendar {
  std::bitset<7> week_days_;
  boost::gregorian::date first_day_, last_day_;
};

std::map<std::string, calendar> read_calendar(loaded_file);

}  // namespace motis::loader::gtfs
