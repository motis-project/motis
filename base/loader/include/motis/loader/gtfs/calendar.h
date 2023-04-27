#pragma once

#include <bitset>
#include <map>
#include <string>

#include "date/date.h"

#include "motis/loader/loaded_file.h"

namespace motis::loader::gtfs {

struct calendar {
  std::bitset<7> week_days_;
  date::sys_days first_day_, last_day_;
};

std::map<std::string, calendar> read_calendar(loaded_file);

}  // namespace motis::loader::gtfs
