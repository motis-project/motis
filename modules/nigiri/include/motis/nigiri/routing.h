#pragma once

#include "motis/module/message.h"

namespace nigiri {
struct timetable;
}

namespace motis::nigiri {

struct tag_lookup;

motis::module::msg_ptr route(tag_lookup const& tags, ::nigiri::timetable& tt,
                             motis::module::msg_ptr const& msg);

}  // namespace motis::nigiri