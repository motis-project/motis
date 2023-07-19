#include "motis/nigiri/extern_trip.h"

#include "nigiri/timetable.h"

#include "motis/nigiri/location.h"
#include "motis/nigiri/unixtime_conv.h"

namespace n = nigiri;

namespace motis::nigiri {

extern_trip nigiri_trip_to_extern_trip(tag_lookup const& tags,
                                       n::timetable const& tt,
                                       n::trip_idx_t const trip_idx,
                                       n::transport const t) {
  auto const resolve_id = [&](n::location_idx_t const x) {
    return get_station_id(
        tags, tt,
        tt.locations_.types_.at(x) == n::location_type::kGeneratedTrack
            ? tt.locations_.parents_.at(x)
            : x);
  };

  auto const [transport, stop_range] =
      tt.trip_transport_ranges_[trip_idx].front();
  auto const first_location = resolve_id(n::stop{
      tt.route_location_seq_[tt.transport_route_[transport]][stop_range.from_]}
                                             .location_idx());
  auto const last_location =
      resolve_id(n::stop{tt.route_location_seq_[tt.transport_route_[transport]]
                                               [stop_range.to_ - 1]}
                     .location_idx());
  auto const section_lines = tt.transport_section_lines_.at(transport);
  auto const line =
      section_lines.empty() ||
              section_lines.front() == n::trip_line_idx_t::invalid()
          ? ""
          : (section_lines.size() == 1
                 ? tt.trip_lines_.at(section_lines.front()).view()
                 : tt.trip_lines_.at(section_lines.at(stop_range.from_))
                       .view());
  auto const x = tt.trip_ids_[trip_idx].at(0);
  return extern_trip{
      .id_ = fmt::format("{}{}", tags.get_tag(tt.trip_id_src_[x]),
                         tt.trip_id_strings_[x].view()),
      .station_id_ = first_location,
      .train_nr_ = tt.trip_train_nr_.at(tt.trip_ids_.at(trip_idx).back()),
      .time_ = to_motis_unixtime(
          tt.event_time(t, stop_range.from_, n::event_type::kDep)),
      .target_station_id_ = last_location,
      .target_time_ = to_motis_unixtime(
          tt.event_time(t, stop_range.to_ - 1, n::event_type::kArr)),
      .line_id_ = std::string{line}};
}

}  // namespace motis::nigiri