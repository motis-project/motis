#include "motis/nigiri/location.h"

#include "utl/to_vec.h"

#include "motis/core/common/logging.h"

#include "nigiri/timetable.h"

namespace n = ::nigiri;

namespace motis::nigiri {

mcd::string get_station_id(tag_lookup const& tags, n::timetable const& tt,
                           n::location_idx_t const l) {
  auto const src = tt.locations_.src_.at(l);
  return fmt::format(
      "{}{}",
      (src == n::source_idx_t::invalid() ? "" : std::string{tags.get_tag(src)}),
      std::string{tt.locations_.ids_.at(l).view()});
}

std::pair<std::string_view, std::string_view> split_tag_and_location_id(
    std::string_view station_id) {
  auto const first_underscore_pos = station_id.find('_');
  return first_underscore_pos != std::string_view::npos
             ? std::pair{station_id.substr(0, first_underscore_pos + 1U),
                         station_id.substr(first_underscore_pos + 1U)}
             : std::pair{std::string_view{}, station_id};
}

n::location_id motis_station_to_nigiri_id(tag_lookup const& tags,
                                          std::string_view station_id) {
  auto const [tag, id] = split_tag_and_location_id(station_id);
  return {n::string{id}, tags.get_src(tag)};
}

n::location_idx_t get_location_idx(tag_lookup const& tags,
                                   n::timetable const& tt,
                                   std::string_view station_id) {
  auto const id = motis_station_to_nigiri_id(tags, station_id);
  try {
    return tt.locations_.location_id_to_idx_.at(id);
  } catch (...) {
    LOG(logging::error) << "nigiri: could not find " << station_id << ", "
                        << id.id_ << ", " << static_cast<int>(to_idx(id.src_))
                        << ", tags: " << tags;
    throw;
  }
}

}  // namespace motis::nigiri