#pragma once

#include <memory>
#include <string_view>
#include <vector>

#include "cista/hash.h"

#include "geo/latlng.h"

#include "motis/protocol/Station_generated.h"

namespace geo {
struct point_rtree;
}

namespace motis {

struct schedule;

struct lookup_station {
  flatbuffers::Offset<Station> to_fbs(flatbuffers::FlatBufferBuilder&) const;
  bool valid() const { return !id_.empty(); }
  std::string_view tag_;
  std::string_view id_;
  std::string_view name_;
  geo::latlng pos_;
};

struct station_lookup {
  station_lookup(std::vector<geo::latlng> const&);
  station_lookup(geo::point_rtree&&);
  station_lookup(station_lookup const&) = delete;
  station_lookup(station_lookup&&) = delete;
  station_lookup& operator=(station_lookup const&) = delete;
  station_lookup& operator=(station_lookup&&) = delete;
  virtual ~station_lookup() noexcept;

  std::size_t size() const;
  std::vector<std::pair<lookup_station, double>> in_radius(
      geo::latlng center, double max_radius) const;
  std::vector<std::size_t> in_radius(geo::latlng center, double min_radius,
                                     double max_radius) const;
  cista::hash_t hash() const;

  virtual lookup_station get(std::size_t idx) const = 0;
  virtual lookup_station get(std::string_view id) const = 0;

private:
  std::unique_ptr<geo::point_rtree> rtree_;
};

struct schedule_station_lookup : public station_lookup {
  schedule_station_lookup(schedule const&);

  schedule_station_lookup(schedule_station_lookup const&) = delete;
  schedule_station_lookup(schedule_station_lookup&&) = delete;
  schedule_station_lookup& operator=(schedule_station_lookup const&) = delete;
  schedule_station_lookup& operator=(schedule_station_lookup&&) = delete;
  virtual ~schedule_station_lookup() noexcept;

  lookup_station get(std::size_t idx) const override;
  lookup_station get(std::string_view id) const override;

private:
  schedule const& sched_;
};

}  // namespace motis