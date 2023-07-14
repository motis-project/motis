#pragma once

#include <ctime>
#include <sstream>

#include "motis/string.h"

#include "motis/core/common/date_time_util.h"
#include "motis/core/common/unixtime.h"

namespace motis {

struct extern_trip {
  CISTA_COMPARABLE()

  std::string to_str() const {
    std::stringstream ss;
    ss << "(id=" << id_ << ", station=" << station_id_
       << ", train_nr=" << train_nr_ << ", time=" << format_unix_time(time_)
       << ", target=" << target_station_id_
       << ", target_time=" << format_unix_time(target_time_)
       << ", line=" << line_id_ << ")";
    return ss.str();
  }

  mcd::string id_;

  mcd::string station_id_;
  uint32_t train_nr_{0};
  unixtime time_{0};

  mcd::string target_station_id_;
  unixtime target_time_{0};
  mcd::string line_id_;
};

}  // namespace motis
