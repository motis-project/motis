#pragma once

#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "motis/module/message.h"
#include "motis/ris/ris_message.h"

namespace motis::ris::risml {

void to_ris_message(std::string_view, std::function<void(ris_message&&)> const&,
                    std::string const& tag = "");
std::vector<ris_message> parse(std::string_view, std::string const& tag = "");

}  // namespace motis::ris::risml
