#pragma once

#include "motis/module/message.h"

#include "nigiri/types.h"

namespace nigiri {
struct timetable;
struct rt_timetable;
}  // namespace nigiri

namespace motis::nigiri {

struct tag_lookup;
struct metrics;

motis::module::msg_ptr route(
    tag_lookup const&, ::nigiri::timetable const&,
    ::nigiri::rt_timetable const*, motis::module::msg_ptr const&, metrics&,
    ::nigiri::profile_idx_t const prf_idx = ::nigiri::profile_idx_t{0U});

}  // namespace motis::nigiri
