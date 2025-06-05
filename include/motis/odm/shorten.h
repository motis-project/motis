#pragma once

namespace motis::odm {

void shorten(std::vector<nigiri::routing::journey>&,
             std::vector<nigiri::routing::start> const& from_rides,
             std::vector<nigiri::routing::start> const& to_rides,
             nigiri::timetable const&,
             nigiri::rt_timetable const*,
             api::plan_params const&);

}  // namespace motis::odm