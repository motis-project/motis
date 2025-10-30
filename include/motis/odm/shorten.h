#pragma once

namespace motis::odm {

void shorten(std::vector<nigiri::routing::journey>& odm_journeys,
             std::vector<nigiri::routing::offset> const& first_mile_taxi,
             std::vector<service_times_t> const& first_mile_taxi_times,
             std::vector<nigiri::routing::offset> const& last_mile_taxi,
             std::vector<service_times_t> const& last_mile_taxi_times,
             nigiri::timetable const&,
             nigiri::rt_timetable const*,
             api::plan_params const&);

}  // namespace motis::odm