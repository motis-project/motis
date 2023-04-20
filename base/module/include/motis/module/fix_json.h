#pragma once

#include <string>
#include <string_view>

#include "motis/module/json_format.h"

namespace motis::module {

struct fix_json_result {
  std::string fixed_json_;
  json_format detected_format_{json_format::DEFAULT_FLATBUFFERS};
};

fix_json_result fix_json(std::string const& json,
                         std::string_view target = std::string_view{});

}  // namespace motis::module
