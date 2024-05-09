#include "motis/core/schedule/station_lookup.h"

#include "utl/to_vec.h"

#include "geo/point_rtree.h"

#include "cista/hashing.h"

#include "motis/core/conv/position_conv.h"

namespace motis {

lookup_station lookup_station::invalid() {
  return lookup_station{"", "", "", geo::latlng{0.0, 0.0}};
}

lookup_station::lookup_station(std::string_view tag, std::string_view id,
                               std::string_view name, geo::latlng pos)
    : tag_{tag}, id_{id}, name_{name}, pos_{pos} {}

flatbuffers::Offset<Station> lookup_station::to_fbs(
    flatbuffers::FlatBufferBuilder& fbb) const {
  auto const pos = motis::to_fbs(pos_);
  return CreateStation(fbb, fbb.CreateString(id()), fbb.CreateString(name_),
                       &pos);
}

std::string lookup_station::id() const {
  return fmt::format("{}{}", tag_, id_);
}

cista::hash_t lookup_station::hash() const {
  auto h = cista::BASE_HASH;
  h = cista::hash_combine(h, cista::hash(id_));
  h = cista::hash_combine(h, cista::hash(tag_));
  h = cista::hash_combine(h, cista::hashing<geo::latlng>{}(pos_));
  return h;
}

geo::latlng lookup_station::pos() const { return pos_; }

station_lookup::~station_lookup() noexcept = default;

station_lookup::station_lookup(geo::point_rtree&& rtree)
    : rtree_{std::make_unique<geo::point_rtree>(std::move(rtree))} {}

cista::hash_t station_lookup::hash() const {
  auto h = cista::BASE_HASH;
  for (auto i = 0U; i != size(); ++i) {
    h = cista::hash_combine(h, cista::hashing<lookup_station>{}(get(i)));
  }
  return h;
}

station_lookup::station_lookup(std::vector<geo::latlng> const& coords)
    : rtree_{
          std::make_unique<geo::point_rtree>(geo::make_point_rtree(coords))} {}

std::size_t station_lookup::size() const { return rtree_->size(); }

std::vector<std::pair<lookup_station, double>> station_lookup::in_radius(
    geo::latlng center, double max_radius) const {
  return utl::to_vec(rtree_->in_radius_with_distance(center, max_radius),
                     [&](std::pair<double, size_t> const x) {
                       return std::pair{get(x.second), x.first};
                     });
}

std::vector<std::size_t> station_lookup::in_radius(
    geo::latlng const center, double const min_radius,
    double const max_radius) const {
  return rtree_->in_radius(center, min_radius, max_radius);
}

}  // namespace motis