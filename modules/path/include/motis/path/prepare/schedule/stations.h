#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>

#include "geo/latlng.h"
#include "geo/point_rtree.h"

#include "utl/erase_duplicates.h"
#include "utl/erase_if.h"
#include "utl/get_or_create.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "motis/path/prepare/schedule/station_sequences.h"

namespace motis::path {

struct station {

  station() = default;

  station(std::string id, std::string name, geo::latlng pos)
      : id_(std::move(id)), name_(std::move(name)), pos_(pos) {}

  friend bool operator<(station const& lhs, station const& rhs) {
    return lhs.id_ < rhs.id_;
  }

  friend bool operator==(station const& lhs, station const& rhs) {
    return lhs.id_ == rhs.id_;
  }

  std::string id_;

  std::string name_;
  std::set<int> categories_;

  geo::latlng pos_;
  std::vector<geo::latlng> stop_positions_;
};

struct station_index {
  explicit station_index(std::vector<station> stations)
      : stations_(std::move(stations)),
        index_(geo::make_point_rtree(stations_,
                                     [](auto const& s) { return s.pos_; })) {}

  std::vector<station> stations_;
  geo::point_rtree index_;
};

inline std::vector<station> collect_stations(
    std::vector<station_seq> const& seqs) {
  std::map<std::string, station> stations;
  for (auto const& seq : seqs) {
    utl::verify(seq.station_ids_.size() == seq.coordinates_.size(),
                "invalid seq!");
    for (auto i = 0UL; i < seq.station_ids_.size(); ++i) {

      utl::get_or_create(stations, seq.station_ids_[i], [&seq, &i] {
        return station{seq.station_ids_[i],  //
                       seq.station_names_[i],  //
                       seq.coordinates_[i]};
      }).categories_.insert(begin(seq.categories_), end(seq.categories_));
    }
  }

  return utl::to_vec(stations, [](auto const& pair) { return pair.second; });
}

inline station_index make_station_index(std::vector<station_seq> const& seqs,
                                        std::vector<station> stations) {
  std::vector<std::string> ids;
  for (auto const& seq : seqs) {
    for (auto const& station_id : seq.station_ids_) {
      ids.emplace_back(station_id);
    }
  }
  utl::erase_duplicates(ids);

  utl::erase_if(stations, [&](auto const& s) {
    auto const it = std::lower_bound(begin(ids), end(ids), s.id_);
    return it == end(ids) || *it != s.id_;
  });

  return station_index{std::move(stations)};
}

}  // namespace motis::path
