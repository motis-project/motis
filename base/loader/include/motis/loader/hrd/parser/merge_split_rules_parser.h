#pragma once

#include <cinttypes>
#include <map>

#include "motis/schedule/bitfield.h"
#include "motis/loader/hrd/model/service_rule.h"
#include "motis/loader/hrd/parse_config.h"
#include "motis/loader/loaded_file.h"

namespace motis::loader::hrd {

void parse_merge_split_service_rules(loaded_file const&,
                                     std::map<int, bitfield> const&,
                                     service_rules&, config const&);

}  // namespace motis::loader::hrd
