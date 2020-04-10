#pragma once

#include "motis/module/message.h"
#include "motis/bikesharing/database.h"
#include "motis/bikesharing/geo_index.h"

namespace motis::bikesharing {

module::msg_ptr geo_terminals(database const&, geo_index const&,
                              BikesharingGeoTerminalsRequest const*);

}  // namespace motis::bikesharing
