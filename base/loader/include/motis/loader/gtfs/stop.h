#pragma once

#include <map>
#include <memory>
#include <string>

#include "motis/loader/loaded_file.h"

namespace motis::loader::gtfs {

struct stop {
  stop(std::string id, std::string name, double lat, double lng,
       std::string timezone)
      : id_(std::move(id)),
        name_(std::move(name)),
        lat_(lat),
        lng_(lng),
        timezone_{std::move(timezone)} {}

  std::string id_;
  std::string name_;
  double lat_, lng_;
  std::string timezone_;
};

using stop_map = std::map<std::string, std::unique_ptr<stop>>;

stop_map read_stops(loaded_file);

}  // namespace motis::loader::gtfs
