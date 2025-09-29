#pragma once

#include "nigiri/types.h"

#include "motis-api/motis-api.h"

namespace motis {

api::ModeEnum to_mode(nigiri::clasz, unsigned api_version);

}  // namespace motis