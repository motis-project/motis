#pragma once

#include <map>
#include <string>
#include <vector>

#include "cista/reflection/comparable.h"

#include "date/date.h"

#include "motis/loader/loaded_file.h"

namespace motis::loader::gtfs {

struct calendar_date {
  CISTA_COMPARABLE()
  enum { ADD, REMOVE } type_{ADD};
  date::sys_days day_;
};

std::map<std::string, std::vector<calendar_date>> read_calendar_date(
    loaded_file);

}  // namespace motis::loader::gtfs
