#pragma once

#include "motis/loader/hrd/model/timezones.h"
#include "motis/loader/hrd/parse_config.h"
#include "motis/loader/loaded_file.h"

namespace motis::loader::hrd {

timezones parse_timezones(loaded_file const&, loaded_file const&,
                          config const&);

}  // namespace motis::loader::hrd
