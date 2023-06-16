#include "motis/nigiri/location.h"

#include "motis/core/common/logging.h"

#include "utl/helpers/algorithm.h"

#include "nigiri/timetable.h"

namespace n = ::nigiri;

namespace motis::nigiri {

n::location_id motis_station_to_nigiri_id(std::vector<std::string> const& tags,
                                          std::string_view station_id) {
  auto const start_tag_it = utl::find_if(
      tags, [&](auto&& tag) { return station_id.starts_with(tag); });
  return start_tag_it == end(tags)
             ? n::location_id{station_id, n::source_idx_t::invalid()}
             : n::location_id{
                   station_id.substr(start_tag_it->length()),
                   n::source_idx_t{static_cast<cista::base_t<n::source_idx_t>>(
                       std::distance(begin(tags), start_tag_it))}};
}

n::location_idx_t get_location_idx(std::vector<std::string> const& tags,
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