#pragma once

#include <map>
#include <string>

#include "ppr/common/routing_graph.h"

#include "motis/ppr/profile_info.h"

namespace motis::ppr {

struct ppr_data {
  ::ppr::routing_graph rg_;
  std::map<std::string, profile_info> profiles_;
};

}  // namespace motis::ppr
