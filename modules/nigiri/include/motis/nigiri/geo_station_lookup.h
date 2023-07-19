#pragma once

#include "geo/point_rtree.h"

#include "motis/core/schedule/station_lookup.h"
#include "motis/module/message.h"
#include "motis/nigiri/tag_lookup.h"

namespace nigiri {
struct timetable;
}

namespace motis::nigiri {

motis::module::msg_ptr geo_station_lookup(station_lookup const&,
                                          motis::module::msg_ptr const&);

motis::module::msg_ptr station_location(tag_lookup const& tags,
                                        ::nigiri::timetable const&,
                                        motis::module::msg_ptr const&);

}  // namespace motis::nigiri