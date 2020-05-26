#pragma once

#include <map>
#include <memory>
#include <set>
#include <string>

#include "geo/latlng.h"

#include "motis/loader/loaded_file.h"

namespace motis::loader::gtfs {

struct stop {
  stop(std::string id, std::string name, double lat, double lng,
       std::string timezone)
      : id_{std::move(id)},
        name_{std::move(name)},
        coord_{lat, lng},
        timezone_{std::move(timezone)} {}

  std::string id_;
  std::string name_;
  geo::latlng coord_;
  std::string timezone_;
  std::set<stop*> same_name_;
};

using stop_map = std::map<std::string, std::unique_ptr<stop>>;

stop_map read_stops(loaded_file);

}  // namespace motis::loader::gtfs
