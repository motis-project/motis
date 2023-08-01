#pragma once

#include "motis/module/message.h"

#include "nigiri/types.h"

namespace nigiri {
struct timetable;
struct rt_timetable;
}  // namespace nigiri

namespace motis::nigiri {

struct tag_lookup;

motis::module::msg_ptr route(
    tag_lookup const&, ::nigiri::timetable const&,
    ::nigiri::rt_timetable const*,
    ::nigiri::vector_map<::nigiri::route_idx_t, std::uint32_t> const&,
    motis::module::msg_ptr const&);

}  // namespace motis::nigiri