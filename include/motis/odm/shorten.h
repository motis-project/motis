#pragma once

namespace motis::odm {

void shorten(std::vector<nigiri::routing::journey>&,
             std::vector<nigiri::routing::start> const& from_rides,
             std::vector<nigiri::routing::start> const& to_rides,
             std::vector<nigiri::routing::offset> const& start_walk,
             std::vector<nigiri::routing::offset> const& dest_walk,
             nigiri::timetable const&,
             nigiri::rt_timetable const*,
             api::plan_params const&);

}  // namespace motis::odm