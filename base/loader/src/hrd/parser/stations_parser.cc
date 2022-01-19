#include "motis/loader/hrd/parser/stations_parser.h"

#include <map>

#include "utl/parser/arg_parser.h"

#include "motis/core/common/logging.h"
#include "motis/loader/parser_error.h"
#include "motis/loader/util.h"

using namespace utl;
using namespace motis::logging;

namespace motis::loader::hrd {

void parse_station_names(loaded_file const& file,
                         std::map<int, intermediate_station>& stations,
                         config const& c) {
  scoped_timer timer("parsing station names");
  for_each_line_numbered(file.content(), [&](cstr line, int line_number) {
    if (line.len == 0 || line[0] == '%') {
      return;
    } else if (line.len < 13) {
      throw parser_error(file.name(), line_number);
    }

    auto eva_num = parse<int>(line.substr(c.st_.names_.eva_));
    auto name = line.substr(c.st_.names_.name_);

    auto it = std::find(begin(name), end(name), '$');
    if (it != end(name)) {
      name.len = std::distance(begin(name), it);
    }

    stations[eva_num].name_ = std::string(name.str, name.len);
  });
}

void parse_station_coordinates(loaded_file const& file,
                               std::map<int, intermediate_station>& stations,
                               config const& c) {
  scoped_timer timer("parsing station coordinates");
  for_each_line_numbered(file.content(), [&](cstr line, int line_number) {
    if (line.len == 0 || line[0] == '%') {
      return;
    } else if (line.len < 30) {
      throw parser_error(file.name(), line_number);
    }

    auto eva_num = parse<int>(line.substr(c.st_.coords_.eva_));
    auto& station = stations[eva_num];

    station.lng_ = parse<double>(line.substr(c.st_.coords_.lng_).trim());
    station.lat_ = parse<double>(line.substr(c.st_.coords_.lat_).trim());
  });
}

void set_change_times(station_meta_data const& metas,
                      std::map<int, intermediate_station>& stations) {
  scoped_timer timer("set station change times");
  for (auto& station_entry : stations) {
    auto const ct = metas.get_station_change_time(station_entry.first);
    station_entry.second.change_time_ = ct.first;
    station_entry.second.platform_change_time_ = ct.second;
  }
}

void set_ds100(station_meta_data const& metas,
               std::map<int, intermediate_station>& stations) {
  for (auto const& ds100 : metas.ds100_to_eva_num_) {
    auto it = stations.find(ds100.second);
    if (it != end(stations)) {
      it->second.ds100_.push_back(ds100.first.to_str());
    }
  }
}

void set_platforms(station_meta_data const& metas,
                   std::map<int, intermediate_station>& stations) {
  for (auto const& platforms : metas.platforms_) {
    auto const ds100_it = metas.ds100_to_eva_num_.find(platforms.first.c_str());
    if (ds100_it == end(metas.ds100_to_eva_num_)) {
      continue;
    }
    auto const eva = ds100_it->second;
    auto station_it = stations.find(eva);
    if (station_it == end(stations)) {
      continue;
    }
    auto& station = station_it->second;
    station.platforms_ = platforms.second;
  }
}

std::map<int, intermediate_station> parse_stations(
    loaded_file const& station_names_file,
    loaded_file const& station_coordinates_file, station_meta_data const& metas,
    config const& config) {
  scoped_timer timer("parsing stations");
  std::map<int, intermediate_station> stations;
  parse_station_names(station_names_file, stations, config);
  parse_station_coordinates(station_coordinates_file, stations, config);
  set_change_times(metas, stations);
  set_ds100(metas, stations);
  set_platforms(metas, stations);
  return stations;
}

}  // namespace motis::loader::hrd
