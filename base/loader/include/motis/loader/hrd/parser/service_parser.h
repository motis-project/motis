#pragma once

#include <cinttypes>
#include <functional>
#include <map>
#include <vector>

#include "motis/loader/hrd/model/hrd_service.h"
#include "motis/loader/hrd/model/specification.h"
#include "motis/loader/hrd/parse_config.h"
#include "motis/loader/loaded_file.h"

namespace motis::loader::hrd {

void parse_specification(loaded_file const&,
                         std::function<void(specification const&)>,
                         std::function<void(std::size_t)> bytes_consumed);

void for_each_service(loaded_file const&, std::map<int, bitfield> const&,
                      std::function<void(hrd_service const&)>,
                      std::function<void(std::size_t)> bytes_consumed,
                      config const&);

}  // namespace motis::loader::hrd
