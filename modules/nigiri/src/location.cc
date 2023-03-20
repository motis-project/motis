#include "motis/nigiri/location.h"

#include "utl/helpers/algorithm.h"

#include "nigiri/timetable.h"

namespace n = ::nigiri;

namespace motis::nigiri {

n::location_id motis_station_to_nigiri_id(std::vector<std::string> const& tags,
                                          std::string const& station_id) {
  auto const it = utl::find_if(
      tags, [&](auto&& tag) { return station_id.starts_with(tag); });
  return it == end(tags)
             ? n::location_id{station_id, n::source_idx_t{0U}}
             : n::location_id{
                   station_id.substr(it->length()),
                   n::source_idx_t{static_cast<cista::base_t<n::source_idx_t>>(
                       std::distance(begin(tags), it))}};
}

n::location_idx_t get_location_idx(std::vector<std::string> const& tags,
                                   n::timetable const& tt,
                                   std::string const& station_id) {
  return tt.locations_.location_id_to_idx_.at(
      motis_station_to_nigiri_id(tags, station_id));
}

}  // namespace motis::nigiri