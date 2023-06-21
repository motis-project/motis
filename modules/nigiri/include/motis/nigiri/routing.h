#pragma once

#include "motis/module/message.h"

namespace nigiri {
struct timetable;
struct rt_timetable;
}  // namespace nigiri

namespace motis::nigiri {

struct tag_lookup;

motis::module::msg_ptr route(tag_lookup const&, ::nigiri::timetable&,
                             ::nigiri::rt_timetable const*,
                             motis::module::msg_ptr const&);

}  // namespace motis::nigiri