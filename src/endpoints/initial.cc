#include "motis/endpoints/initial.h"
#include "motis/config.h"

#include "utl/erase_if.h"
#include "utl/to_vec.h"

#include "tiles/fixed/convert.h"
#include "tiles/fixed/fixed_geometry.h"

#include "nigiri/timetable.h"

namespace motis::ep {

api::initial_response initial::operator()(boost::urls::url_view const&) const {

  auto const limits = config_.get_limits();
  return {
      .lat_ = tt_.bbox_.center_.lat_,
      .lon_ = tt_.bbox_.center_.lng_,
      .zoom_ = tt_.bbox_.zoom_,
      .serverConfig_ = api::ServerConfig{
          .motisVersion_ = std::string{motis_version_},
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
              static_cast<double>(limits.street_routing_max_direct_seconds_),
          .shapesDebugEnabled_ = config_.shapes_debug_api_enabled()}};
}

}  // namespace motis::ep
