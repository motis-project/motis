#pragma once

#include <string>
#include <vector>

#include "motis/module/message.h"

namespace nigiri {
struct timetable;
}

namespace motis::nigiri {

motis::module::msg_ptr route(std::vector<std::string> const& tags,
                             ::nigiri::timetable& tt,
                             motis::module::msg_ptr const& msg);

}  // namespace motis::nigiri