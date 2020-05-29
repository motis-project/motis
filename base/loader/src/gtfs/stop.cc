#include "motis/loader/gtfs/stop.h"

#include <algorithm>
#include <string>
#include <tuple>

#include "utl/get_or_create.h"
#include "utl/parser/csv.h"

#include "motis/core/common/logging.h"
#include "motis/hash_map.h"

using namespace utl;
using std::get;

namespace motis::loader::gtfs {

enum { stop_id, stop_name, stop_timezone, parent_station, stop_lat, stop_lon };
using gtfs_stop = std::tuple<cstr, cstr, cstr, cstr, float, float>;
static const column_mapping<gtfs_stop> columns = {
    {"stop_id", "stop_name", "stop_timezone", "parent_station", "stop_lat",
     "stop_lon"}};

std::set<stop*> stop::get_metas(std::vector<stop*> const& stops,
                                geo::point_rtree const& stop_rtree) {
  std::set<stop*> todo, done;
  todo.emplace(this);
  todo.insert(begin(same_name_), end(same_name_));
  for (auto const& idx : stop_rtree.in_radius(coord_, 100)) {
    todo.insert(stops[idx]);
  }

  while (!todo.empty()) {
    auto const next = *todo.begin();
    todo.erase(todo.begin());
    done.emplace(next);

    for (auto const& p : next->parents_) {
      if (done.find(p) == end(done)) {
        todo.emplace(p);
      }
    }

    for (auto const& p : next->children_) {
      if (done.find(p) == end(done)) {
        todo.emplace(p);
      }
    }
  }

  for (auto it = begin(done); it != end(done);) {
    auto* meta = *it;
    auto const is_parent = parents_.find(meta) != end(parents_);
    auto const is_child = children_.find(meta) != end(children_);
    auto const distance_in_m = geo::distance(meta->coord_, coord_);
    if ((distance_in_m > 500 && !is_parent && !is_child) ||
        distance_in_m > 2000) {
      it = done.erase(it);
    } else {
      ++it;
    }
  }

  return done;
}

stop_map read_stops(loaded_file file) {
  motis::logging::scoped_timer timer{"read stops"};

  stop_map stops;
  mcd::hash_map<std::string, std::vector<stop*>> equal_names;
  for (auto const& s : read<gtfs_stop>(file.content(), columns)) {
    auto const new_stop =
        utl::get_or_create(stops, get<stop_id>(s).to_str(), [&]() {
          return std::make_unique<stop>();
        }).get();

    new_stop->id_ = get<stop_id>(s).to_str();
    new_stop->name_ = get<stop_name>(s).to_str();
    new_stop->coord_ = {get<stop_lat>(s), get<stop_lon>(s)};
    new_stop->timezone_ = get<stop_timezone>(s).to_str();

    if (!get<parent_station>(s).trim().empty()) {
      auto const parent =
          utl::get_or_create(stops, get<parent_station>(s).trim().to_str(),
                             []() { return std::make_unique<stop>(); })
              .get();
      parent->id_ = get<parent_station>(s).trim().to_str();
      parent->children_.emplace(new_stop);
      new_stop->parents_.emplace(parent);
    }

    equal_names[get<stop_name>(s).view()].emplace_back(new_stop);
  }

  for (auto const& [id, s] : stops) {
    for (auto const& equal : equal_names[s->name_]) {
      if (equal != s.get()) {
        s->same_name_.emplace(equal);
      }
    }
  }

  return stops;
}

}  // namespace motis::loader::gtfs
