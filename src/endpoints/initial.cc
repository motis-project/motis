#include "motis/endpoints/initial.h"
#include "motis/config.h"

#include "utl/erase_if.h"
#include "utl/to_vec.h"

#include "tiles/fixed/convert.h"
#include "tiles/fixed/fixed_geometry.h"

#include "nigiri/timetable.h"

namespace n = nigiri;

namespace motis::ep {

api::initial_response initial::operator()(boost::urls::url_view const&) const {
  auto const get_quantiles = [](std::vector<double>&& coords) {
    utl::erase_if(coords, [](auto const c) { return c == 0.; });
    if (coords.empty()) {
      return std::make_pair(0., 0.);
    }
    if (coords.size() < 10) {
      return std::make_pair(coords.front(), coords.back());
    }

    std::sort(begin(coords), end(coords));
    constexpr auto const kQuantile = .8;
    return std::make_pair(
        coords.at(static_cast<double>(coords.size()) * (1 - kQuantile)),
        coords.at(static_cast<double>(coords.size()) * (kQuantile)));
  };

  auto zoom = 0U;
  auto center = geo::latlng{};

  if (tt_ != nullptr) {
    auto const [lat_min, lat_max] = get_quantiles(utl::to_vec(
        tt_->locations_.coordinates_, [](auto const& s) { return s.lat_; }));
    auto const [lng_min, lng_max] = get_quantiles(utl::to_vec(
        tt_->locations_.coordinates_, [](auto const& s) { return s.lng_; }));

    auto const fixed0 = tiles::latlng_to_fixed({lat_min, lng_min});
    auto const fixed1 = tiles::latlng_to_fixed({lat_max, lng_max});

    center = tiles::fixed_to_latlng(
        {(fixed0.x() + fixed1.x()) / 2, (fixed0.y() + fixed1.y()) / 2});

    auto const d = static_cast<unsigned>(std::max(
        std::abs(fixed0.x() - fixed1.x()), std::abs(fixed0.y() - fixed1.y())));

    for (; zoom < (tiles::kMaxZoomLevel - 1); ++zoom) {
      if (((tiles::kTileSize * 2ULL) *
           (1ULL << (tiles::kMaxZoomLevel - (zoom + 1)))) < d) {
        break;
      }
    }
  }

  auto const limits = config_.get_limits();
  return {
      .lat_ = center.lat_,
      .lon_ = center.lng_,
      .zoom_ = static_cast<double>(zoom),
      .serverConfig_ = api::ServerConfig{
          .hasElevation_ = config_.get_street_routing()
                               .transform([](config::street_routing const& x) {
                                 return x.elevation_data_dir_.has_value();
                               })
                               .value_or(false),
          .hasRoutedTransfers_ = config_.osr_footpath_,
          .hasStreetRouting_ = config_.get_street_routing().has_value(),
          .maxOneToManySize_ = static_cast<double>(limits.onetomany_max_many_),
          .maxOneToAllTravelTimeLimit_ =
              static_cast<double>(limits.onetoall_max_travel_minutes_),
          .maxPrePostTransitTimeLimit_ = static_cast<double>(
              limits.street_routing_max_prepost_transit_seconds_),
          .maxDirectTimeLimit_ =
              static_cast<double>(limits.street_routing_max_direct_seconds_)}};
}

}  // namespace motis::ep
