#pragma once

#include "motis/module/message.h"
#include "motis/bikesharing/database.h"
#include "motis/bikesharing/geo_index.h"

namespace motis::bikesharing {

module::msg_ptr find_connections(database const&, geo_index const&,
                                 BikesharingRequest const*);

}  // namespace motis::bikesharing
