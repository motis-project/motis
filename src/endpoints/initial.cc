#include "motis/endpoints/initial.h"
#include "motis/config.h"

#include "utl/erase_if.h"
#include "utl/to_vec.h"

#include "tiles/fixed/convert.h"
#include "tiles/fixed/fixed_geometry.h"

#include "nigiri/timetable.h"

namespace motis::ep {

api::initial_response initial::operator()(boost::urls::url_view const&) const {

  auto const onetoall_max_travel_minutes =
      config_.limits_->onetoall_max_travel_minutes_;
  auto const street_routing_max_direct_seconds =
      config_.limits_->street_routing_max_direct_seconds_;
  auto const street_routing_max_prepost_transit_seconds =
      config_.limits_->street_routing_max_prepost_transit_seconds_;
  auto const has_osr_footpath = config_.osr_footpath_;
  auto const has_street_routing = config_.get_street_routing().has_value();
  auto const has_elevation_data_dir =
      has_street_routing &&
      config_.get_street_routing()->elevation_data_dir_.has_value();

  return {.lat_ = tt_.bbox_.center_.lat_,
          .lon_ = tt_.bbox_.center_.lng_,
          .zoom_ = tt_.bbox_.zoom_,
          .serverConfig_ = api::ServerConfig{
              .hasElevation_ = has_elevation_data_dir,
              .hasRoutedTransfers_ = has_osr_footpath,
              .hasStreetRouting_ = has_street_routing,
              .maxOneToAllTravelTimeLimit_ =
                  static_cast<double>(onetoall_max_travel_minutes),
              .maxPrePostTransitTimeLimit_ = static_cast<double>(
                  street_routing_max_prepost_transit_seconds),
              .maxDirectTimeLimit_ =
                  static_cast<double>(street_routing_max_direct_seconds)}};
}

}  // namespace motis::ep
