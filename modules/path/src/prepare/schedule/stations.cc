#include "motis/path/prepare/schedule/stations.h"

#include "utl/equal_ranges_linear.h"
#include "utl/get_or_create.h"
#include "utl/pairwise.h"
#include "utl/to_vec.h"

#include "motis/hash_map.h"
#include "motis/hash_set.h"

#include "motis/path/prepare/tuning_parameters.h"

namespace motis::path {

std::vector<station> collect_stations(mcd::vector<station_seq> const& seqs) {
  mcd::hash_map<mcd::string, station> stations;
  for (auto const& seq : seqs) {
    utl::verify(seq.station_ids_.size() == seq.coordinates_.size(),
                "invalid seq!");
    for (auto i = 0ULL; i < seq.station_ids_.size(); ++i) {
      utl::get_or_create(stations, seq.station_ids_[i], [&] {
        return station{seq.station_ids_[i].str(),  //
                       seq.station_names_[i].str(),  //
                       seq.coordinates_[i]};
      }).classes_.insert(begin(seq.classes_), end(seq.classes_));
    }
  }

  return utl::to_vec(stations, [](auto const& pair) { return pair.second; });
}

station_index make_station_index(std::vector<station> stations) {
  station_index result;
  result.stations_ = std::move(stations);
  result.index_ = geo::make_point_rtree(result.stations_,
                                        [](auto const& s) { return s.pos_; });
  return result;
}

station_index load_stations(mcd::vector<station_seq> const& sequences,
                            osm_data const& data) {
  auto index = make_station_index(collect_stations(sequences));

  mcd::hash_map<std::string, mcd::hash_set<std::string>> links;
  for (auto const& seq : sequences) {
    for (auto const& [a, b] : utl::pairwise(seq.station_ids_)) {
      if (a != b) {
        links[a.str()].insert(b.str());
        links[b.str()].insert(a.str());
      }
    }
  }

  // add to all stations, which have no linked station that is closer
  auto const annotate_stop_positions = [&](auto lb, auto ub) {
    if (std::any_of(lb, ub, [](auto const& sp) {
          return sp.has_category(source_spec::category::BUS);
        })) {
      for (auto it = lb; it != ub; ++it) {
        for (auto const& idx : index.index_.in_radius(it->pos_, 100)) {
          index.stations_.at(idx).stop_positions_.emplace_back(*it);
        }
      }
      return;
    }

    std::map<size_t, double> distances;
    auto const update_distance = [&](auto const idx, auto const distance) {
      auto& stored_distance = utl::get_or_create(distances, idx, [] {
        return std::numeric_limits<double>::infinity();
      });
      stored_distance = std::min(stored_distance, distance);
    };

    for (auto it = lb; it != ub; ++it) {
      for (auto const& [distance, idx] : index.index_.in_radius_with_distance(
               it->pos_, kStationStopDistance)) {
        update_distance(idx, distance);
      }
    }

    // fallback: large radius, take best
    if (distances.empty()) {
      for (auto it = lb; it != ub; ++it) {
        for (auto const& [distance, idx] : index.index_.in_radius_with_distance(
                 it->pos_, kStationStopFallbackDistance)) {
          update_distance(idx, distance);
        }
      }
      if (distances.empty()) {
        return;  // fallback failed
      }
      // XXX revisit this with multiple schedules
      distances = {*std::min_element(
          begin(distances), end(distances),
          [](auto&& a, auto&& b) { return a.second < b.second; })};
    }

    for (auto const& self : distances) {
      auto const link_it = links.find(index.stations_.at(self.first).id_);
      if (link_it != end(links) &&
          std::any_of(begin(distances), end(distances), [&](auto const& other) {
            auto const other_it =
                link_it->second.find(index.stations_.at(other.first).id_);
            return other_it != end(link_it->second) &&
                   other.second < self.second;
          })) {
        continue;  // other is closer than self
      }

      for (auto it = lb; it != ub; ++it) {
        index.stations_.at(self.first).stop_positions_.push_back(*it);
      }
    }
  };

  auto const sp_begin = begin(data.stop_positions_);
  auto const sp_end = end(data.stop_positions_);
  auto const name_less = [](auto&& a, auto&& b) { return a.name_ < b.name_; };
  auto const name_eq = [](auto&& a, auto&& b) { return a.name_ == b.name_; };

  utl::verify(std::is_sorted(sp_begin, sp_end, name_less),
              "stop positions not sorted");

  auto const first_named = std::find_if(
      sp_begin, sp_end, [](auto&& sp) { return !sp.name_.empty(); });
  for (auto it = sp_begin; it != first_named; ++it) {
    annotate_stop_positions(it, std::next(it));
  }
  utl::equal_ranges_linear(first_named, sp_end, name_eq,
                           annotate_stop_positions);

  return index;
}
}  // namespace motis::path
