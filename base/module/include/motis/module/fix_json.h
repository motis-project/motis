#pragma once

#include <string>
#include <string_view>

namespace motis::module {

std::string fix_json(std::string const& json,
                     std::string_view target = std::string_view{});

}  // namespace motis::module
