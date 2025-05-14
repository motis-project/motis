#pragma once

#include "motis/fwd.h"

namespace motis::gbfs {

struct gbfs_provider;
struct provider_routing_data;

void map_data(osr::ways const&,
              osr::lookup const&,
              gbfs_provider const&,
              provider_routing_data&);

}  // namespace motis::gbfs
