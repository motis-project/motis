#pragma once

#include <map>
#include <set>

#include "motis/loader/hrd/parse_config.h"
#include "motis/loader/hrd/parser/station_meta_data_parser.h"
#include "motis/loader/loaded_file.h"
#include "motis/schedule-format/Station_generated.h"

namespace motis::loader::hrd {

struct intermediate_station {
  std::string name_;
  int change_time_{0};
  int platform_change_time_{0};
  double lng_{0.0}, lat_{0.0};
  std::vector<std::string> ds100_;
  std::map<std::string, std::set<std::string>> platforms_;
};

std::map<int, intermediate_station> parse_stations(
    loaded_file const& station_names_file,
    loaded_file const& station_coordinates_file, station_meta_data const&,
    config const&);

}  // namespace motis::loader::hrd
