#include "motis/path/prepare/schedule/stations.h"

#include "utl/get_or_create.h"
#include "utl/to_vec.h"

#include "motis/hash_map.h"

namespace motis::path {

station_index collect_stations(mcd::vector<station_seq> const& seqs) {
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

  return make_station_index(
      utl::to_vec(stations, [](auto const& pair) { return pair.second; }));
}

station_index make_station_index(std::vector<station> stations) {
  station_index result;
  result.stations_ = std::move(stations);
  result.index_ = geo::make_point_rtree(result.stations_,
                                        [](auto const& s) { return s.pos_; });
  return result;
}

void annotate_stop_positions(osm_data const& data, station_index& stations) {
  for (auto const& pos : data.stop_positions_) {
    for (auto const& idx : stations.index_.in_radius(pos, 100)) {
      stations.stations_.at(idx).stop_positions_.emplace_back(pos);
      break;  // XXX revisit this with multiple schedules
    }
  }
}

}  // namespace motis::path
