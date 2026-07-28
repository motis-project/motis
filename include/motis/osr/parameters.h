#pragma once

#include <cstdint>

#include "osr/routing/parameters.h"
#include "osr/routing/profile.h"

#include "motis-api/motis-api.h"

namespace motis {

struct osr_parameters {
  constexpr static auto const kFootSpeed = 1.2F;
  constexpr static auto const kWheelchairSpeed = 0.8F;
  constexpr static auto const kBikeSpeed = 4.2F;
  constexpr static auto const kCarSpeed = 28.0F;
  constexpr static auto const kBusSpeed = 28.0F;
  constexpr static auto const kRailwaySpeed = 28.0F;
  constexpr static auto const kFerrySpeed = 28.0F;
  constexpr static auto const kHgvHeightMeters = 4.0F;
  constexpr static auto const kHgvWidthMeters = 2.55F;
  constexpr static auto const kHgvLengthMeters = 18.75F;
  constexpr static auto const kHgvWeightTons = 40.0F;
  constexpr static auto const kHgvHazmat = false;
  constexpr static auto const kHgvHazmatWater = false;
  constexpr static auto const kHgvAxleCount = std::uint8_t{5U};
  constexpr static auto const kHgvAxleLoadTons = 11.5F;
  constexpr static auto const kHgvTrailer = true;
  constexpr static auto const kHgvTopSpeedKmh = std::uint8_t{80U};
  constexpr static auto const kHgvLowEmissionZoneAccess = true;

  float pedestrian_speed_{kFootSpeed};
  float cycling_speed_{kBikeSpeed};
  bool use_wheelchair_{false};
  float hgv_height_meters_{kHgvHeightMeters};
  float hgv_width_meters_{kHgvWidthMeters};
  float hgv_length_meters_{kHgvLengthMeters};
  float hgv_weight_tons_{kHgvWeightTons};
  bool hgv_hazmat_{kHgvHazmat};
  bool hgv_hazmat_water_{kHgvHazmatWater};
  std::uint8_t hgv_axle_count_{kHgvAxleCount};
  float hgv_axle_load_tons_{kHgvAxleLoadTons};
  bool hgv_trailer_{kHgvTrailer};
  std::uint8_t hgv_top_speed_km_h_{kHgvTopSpeedKmh};
  bool hgv_low_emission_zone_access_{kHgvLowEmissionZoneAccess};
};

osr_parameters get_osr_parameters(api::plan_params const&);

osr_parameters get_osr_parameters(api::refreshItinerary_params const&);

osr_parameters get_osr_parameters(api::oneToAll_params const&);

osr_parameters get_osr_parameters(api::oneToMany_params const&);

osr_parameters get_osr_parameters(api::OneToManyParams const&);

osr_parameters get_osr_parameters(api::oneToManyIntermodal_params const&);

osr_parameters get_osr_parameters(api::OneToManyIntermodalParams const&);

osr::profile_parameters to_profile_parameters(osr::search_profile,
                                              osr_parameters const&);
}  // namespace motis
