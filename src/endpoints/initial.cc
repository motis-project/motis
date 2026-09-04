#include "motis/endpoints/initial.h"
#include "motis/config.h"

#include "utl/erase_if.h"
#include "utl/to_vec.h"

#include "tiles/fixed/convert.h"
#include "tiles/fixed/fixed_geometry.h"

#include "geo/box.h"

#include "nigiri/timetable.h"

#include "motis/data.h"

namespace n = nigiri;

namespace motis::ep {

std::pair<geo::latlng, unsigned> get_center_and_zoom(geo::latlng const& min,
                                                     geo::latlng const& max) {
  auto const fixed0 = tiles::latlng_to_fixed(min);
  auto const fixed1 = tiles::latlng_to_fixed(max);

  auto const center = tiles::fixed_to_latlng(
      {(fixed0.x() + fixed1.x()) / 2, (fixed0.y() + fixed1.y()) / 2});

  auto zoom = 0U;
  auto const span = static_cast<unsigned>(std::max(
      std::abs(fixed0.x() - fixed1.x()), std::abs(fixed0.y() - fixed1.y())));

  for (; zoom < (tiles::kMaxZoomLevel - 1); ++zoom) {
    if (((tiles::kTileSize * 2ULL) *
         (1ULL << (tiles::kMaxZoomLevel - (zoom + 1)))) < span) {
      break;
    }
  }

  return {center, zoom};
}

std::optional<std::pair<geo::latlng, unsigned>> get_osr_center_and_zoom(
    data const& d) {
  if (d.w_ == nullptr) {
    return std::nullopt;
  }

  auto bbox = geo::box{};

  for (auto node = osr::node_idx_t{0U}; node != d.w_->n_nodes(); ++node) {
    if (d.w_->r_->node_ways_[node].empty()) {
      continue;
    }

    bbox.extend(d.w_->get_node_pos(node).as_latlng());
  }

  if (bbox.empty()) {
    return std::nullopt;
  }

  return {get_center_and_zoom(bbox.min_, bbox.max_)};
}

api::initial_response get_initial_response(data const& d,
                                           std::string_view motis_version) {
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

  auto const tt = d.tt_.get();
  if (tt != nullptr) {
    auto const [lat_min, lat_max] = get_quantiles(utl::to_vec(
        tt->locations_.coordinates_, [](auto const& s) { return s.lat_; }));
    auto const [lng_min, lng_max] = get_quantiles(utl::to_vec(
        tt->locations_.coordinates_, [](auto const& s) { return s.lng_; }));

    std::tie(center, zoom) =
        get_center_and_zoom({lat_min, lng_min}, {lat_max, lng_max});
  } else if (auto const osr_center_zoom = get_osr_center_and_zoom(d);
             osr_center_zoom.has_value()) {
    std::tie(center, zoom) = *osr_center_zoom;
  }

  auto const limits = d.config_.get_limits();
  return {
      .lat_ = center.lat_,
      .lon_ = center.lng_,
      .zoom_ = static_cast<double>(zoom),
      .serverConfig_ = api::ServerConfig{
          .motisVersion_ = std::string{motis_version},
          .hasElevation_ = d.config_.get_street_routing()
                               .transform([](config::street_routing const& x) {
                                 return x.elevation_data_dir_.has_value();
                               })
                               .value_or(false),
          .hasRoutedTransfers_ = d.config_.osr_footpath_,
          .hasStreetRouting_ = d.config_.get_street_routing().has_value(),
          .maxOneToManySize_ = static_cast<double>(limits.onetomany_max_many_),
          .maxOneToAllTravelTimeLimit_ =
              static_cast<double>(limits.onetoall_max_travel_minutes_),
          .maxPrePostTransitTimeLimit_ = static_cast<double>(
              limits.street_routing_max_prepost_transit_seconds_),
          .maxDirectTimeLimit_ =
              static_cast<double>(limits.street_routing_max_direct_seconds_),
          .shapesDebugEnabled_ = d.config_.shapes_debug_api_enabled()}};
}

api::initial_response initial::operator()(boost::urls::url_view const&) const {
  return response_;
}

}  // namespace motis::ep
