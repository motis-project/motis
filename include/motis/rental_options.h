#pragma once

#include <optional>
#include <string>
#include <vector>

#include "motis-api/motis-api.h"

namespace motis {

struct rental_options {
  std::optional<std::vector<api::RentalFormFactorEnum>> form_factors_{};
  std::optional<std::vector<api::RentalPropulsionTypeEnum>> propulsion_types_{};
  std::optional<std::vector<std::string>> providers_{};
  std::optional<std::vector<std::string>> provider_groups_{};
  bool ignore_return_constraints_{false};
};

}  // namespace motis
