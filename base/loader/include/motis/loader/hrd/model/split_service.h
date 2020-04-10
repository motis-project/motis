#pragma once

#include <vector>

#include "motis/loader/hrd/model/hrd_service.h"
#include "motis/loader/hrd/parser/bitfields_parser.h"

namespace motis::loader::hrd {

void expand_traffic_days(hrd_service const&, std::map<int, bitfield> const&,
                         std::vector<hrd_service>&);

}  // namespace motis::loader::hrd
