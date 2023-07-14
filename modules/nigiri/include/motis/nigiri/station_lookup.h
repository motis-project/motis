#pragma once

#include "motis/core/schedule/station_lookup.h"

namespace nigiri {
struct timetable;
}

namespace motis::nigiri {

struct tag_lookup;

struct nigiri_station_lookup : public station_lookup {
  nigiri_station_lookup(tag_lookup const&, ::nigiri::timetable const&);

  nigiri_station_lookup(nigiri_station_lookup const&) = delete;
  nigiri_station_lookup(nigiri_station_lookup&&) = delete;
  nigiri_station_lookup& operator=(nigiri_station_lookup const&) = delete;
  nigiri_station_lookup& operator=(nigiri_station_lookup&&) = delete;
  ~nigiri_station_lookup() noexcept override;

  lookup_station get(std::size_t idx) const override;
  lookup_station get(std::string_view id) const override;

private:
  ::nigiri::timetable const& tt_;
  tag_lookup const& tags_;
};

}  // namespace motis::nigiri