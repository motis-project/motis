#pragma once

#include <optional>
#include <vector>

#include "adr/types.h"

#include "motis-api/motis-api.h"

namespace motis {

adr::filter_type to_filter_type(
    std::optional<std::vector<motis::api::LocationTypeEnum>> const&);

}  // namespace motis