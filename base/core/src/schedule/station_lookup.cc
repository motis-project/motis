#include "motis/core/schedule/station_lookup.h"

#include "utl/to_vec.h"

#include "geo/point_rtree.h"

#include "motis/core/schedule/schedule.h"
#include "motis/core/conv/position_conv.h"

namespace motis {

flatbuffers::Offset<Station> lookup_station::to_fbs(
    flatbuffers::FlatBufferBuilder& fbb) const {
  auto const pos = motis::to_fbs(pos_);
  return CreateStation(fbb, fbb.CreateString(fmt::format("{}{}", tag_, id_)),
                       fbb.CreateString(id_), &pos);
}

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

schedule_station_lookup::schedule_station_lookup(motis::schedule const& sched)
    : station_lookup{utl::to_vec(sched.stations_,
                                 [](auto const& s) {
                                   return geo::latlng{s->lat(), s->lng()};
                                 })},
      sched_{sched} {}

schedule_station_lookup::~schedule_station_lookup() noexcept = default;

lookup_station schedule_station_lookup::get(std::size_t const idx) const {
  auto const& station = *sched_.stations_.at(idx);
  auto s = lookup_station{};
  s.tag_ = "";
  s.id_ = station.eva_nr_;
  s.name_ = station.name_;
  s.pos_ = {station.lat(), station.lng()};
  return s;
}

lookup_station schedule_station_lookup::get(std::string_view id) const {
  return get(sched_.eva_to_station_.at(id)->index_);
}

}  // namespace motis