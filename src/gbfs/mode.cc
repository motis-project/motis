#include "motis/gbfs/mode.h"

#include <utility>

#include "utl/helpers/algorithm.h"
#include "utl/verify.h"

#include "motis/constants.h"

namespace motis::gbfs {

api::RentalFormFactorEnum to_api_form_factor(vehicle_form_factor const ff) {
  switch (ff) {
    case vehicle_form_factor::kBicycle:
      return api::RentalFormFactorEnum::BICYCLE;
    case vehicle_form_factor::kCargoBicycle:
      return api::RentalFormFactorEnum::CARGO_BICYCLE;
    case vehicle_form_factor::kCar: return api::RentalFormFactorEnum::CAR;
    case vehicle_form_factor::kMoped: return api::RentalFormFactorEnum::MOPED;
    case vehicle_form_factor::kScooterStanding:
      return api::RentalFormFactorEnum::SCOOTER_STANDING;
    case vehicle_form_factor::kScooterSeated:
      return api::RentalFormFactorEnum::SCOOTER_SEATED;
    case vehicle_form_factor::kOther: return api::RentalFormFactorEnum::OTHER;
  }
  std::unreachable();
}

vehicle_form_factor from_api_form_factor(api::RentalFormFactorEnum const ff) {
  switch (ff) {
    case api::RentalFormFactorEnum::BICYCLE:
      return vehicle_form_factor::kBicycle;
    case api::RentalFormFactorEnum::CARGO_BICYCLE:
      return vehicle_form_factor::kCargoBicycle;
    case api::RentalFormFactorEnum::CAR: return vehicle_form_factor::kCar;
    case api::RentalFormFactorEnum::MOPED: return vehicle_form_factor::kMoped;
    case api::RentalFormFactorEnum::SCOOTER_STANDING:
      return vehicle_form_factor::kScooterStanding;
    case api::RentalFormFactorEnum::SCOOTER_SEATED:
      return vehicle_form_factor::kScooterSeated;
    case api::RentalFormFactorEnum::OTHER: return vehicle_form_factor::kOther;
  }
  throw utl::fail("invalid rental form factor");
}

api::RentalPropulsionTypeEnum to_api_propulsion_type(propulsion_type const pt) {
  switch (pt) {
    case propulsion_type::kHuman: return api::RentalPropulsionTypeEnum::HUMAN;
    case propulsion_type::kElectricAssist:
      return api::RentalPropulsionTypeEnum::ELECTRIC_ASSIST;
    case propulsion_type::kElectric:
      return api::RentalPropulsionTypeEnum::ELECTRIC;
    case propulsion_type::kCombustion:
      return api::RentalPropulsionTypeEnum::COMBUSTION;
    case propulsion_type::kCombustionDiesel:
      return api::RentalPropulsionTypeEnum::COMBUSTION_DIESEL;
    case propulsion_type::kHybrid: return api::RentalPropulsionTypeEnum::HYBRID;
    case propulsion_type::kPlugInHybrid:
      return api::RentalPropulsionTypeEnum::PLUG_IN_HYBRID;
    case propulsion_type::kHydrogenFuelCell:
      return api::RentalPropulsionTypeEnum::HYDROGEN_FUEL_CELL;
  }
  std::unreachable();
}

propulsion_type from_api_propulsion_type(
    api::RentalPropulsionTypeEnum const pt) {
  switch (pt) {
    case api::RentalPropulsionTypeEnum::HUMAN: return propulsion_type::kHuman;
    case api::RentalPropulsionTypeEnum::ELECTRIC_ASSIST:
      return propulsion_type::kElectricAssist;
    case api::RentalPropulsionTypeEnum::ELECTRIC:
      return propulsion_type::kElectric;
    case api::RentalPropulsionTypeEnum::COMBUSTION:
      return propulsion_type::kCombustion;
    case api::RentalPropulsionTypeEnum::COMBUSTION_DIESEL:
      return propulsion_type::kCombustionDiesel;
    case api::RentalPropulsionTypeEnum::HYBRID: return propulsion_type::kHybrid;
    case api::RentalPropulsionTypeEnum::PLUG_IN_HYBRID:
      return propulsion_type::kPlugInHybrid;
    case api::RentalPropulsionTypeEnum::HYDROGEN_FUEL_CELL:
      return propulsion_type::kHydrogenFuelCell;
  }
  throw utl::fail("invalid rental propulsion type");
}

api::RentalReturnConstraintEnum to_api_return_constraint(
    return_constraint const rc) {
  switch (rc) {
    case return_constraint::kFreeFloating:
      return api::RentalReturnConstraintEnum::NONE;
    case return_constraint::kAnyStation:
      return api::RentalReturnConstraintEnum::ANY_STATION;
    case return_constraint::kRoundtripStation:
      return api::RentalReturnConstraintEnum::ROUNDTRIP_STATION;
  }
  std::unreachable();
}

bool products_match(
    provider_products const& prod,
    std::optional<std::vector<api::RentalFormFactorEnum>> const& form_factors,
    std::optional<std::vector<api::RentalPropulsionTypeEnum>> const&
        propulsion_types) {
  if (form_factors.has_value() &&
      utl::find(*form_factors, to_api_form_factor(prod.form_factor_)) ==
          end(*form_factors)) {
    return false;
  }
  if (propulsion_types.has_value() &&
      utl::find(*propulsion_types,
                to_api_propulsion_type(prod.propulsion_type_)) ==
          end(*propulsion_types)) {
    return false;
  }
  return true;
}

}  // namespace motis::gbfs
