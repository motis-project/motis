#pragma once

#include <map>
#include <string>
#include <vector>

#include "boost/date_time/gregorian/gregorian.hpp"

#include "motis/loader/bitfield.h"
#include "motis/loader/gtfs/calendar.h"
#include "motis/loader/gtfs/calendar_date.h"

namespace motis::loader::gtfs {

struct traffic_days {
  boost::gregorian::date first_day_, last_day_;
  std::map<std::string, std::unique_ptr<bitfield>> traffic_days_;
};

traffic_days merge_traffic_days(
    std::map<std::string, calendar> const&,
    std::map<std::string, std::vector<calendar_date>> const&);

}  // namespace motis::loader::gtfs
