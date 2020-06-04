#pragma once

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "geo/latlng.h"
#include "geo/point_rtree.h"

#include "motis/loader/loaded_file.h"

namespace motis::loader::gtfs {

struct stop {
  std::set<stop*> get_metas(std::vector<stop*> const&,
                            geo::point_rtree const& stop_rtree);

  std::string id_;
  std::string name_;
  geo::latlng coord_;
  std::string timezone_;
  std::set<stop*> same_name_, parents_, children_;
};

using stop_map = std::map<std::string, std::unique_ptr<stop>>;

stop_map read_stops(loaded_file);

}  // namespace motis::loader::gtfs
