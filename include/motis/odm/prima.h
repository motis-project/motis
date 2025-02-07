#pragma once

#include <chrono>
#include <vector>

#include "geo/latlng.h"

#include "nigiri/routing/journey.h"
#include "nigiri/routing/start_times.h"

#include "motis-api/motis-api.h"

namespace motis::ep {
struct routing;
}  // namespace motis::ep

namespace motis::odm {

struct direct_ride {
  nigiri::unixtime_t dep_;
  nigiri::unixtime_t arr_;
};

struct capacities {
  std::uint8_t wheelchairs_;
  std::uint8_t bikes_;
  std::uint8_t passengers_;
  std::uint8_t luggage_;
};

struct prima {
  void init(api::Place const& from,
            api::Place const& to,
            api::plan_params const& query);
  std::string get_prima_request(nigiri::timetable const&) const;
  size_t n_events() const;
  bool blacklist_update(std::string_view json);
  bool whitelist_update(std::string_view json);
  void adjust_to_whitelisting();

  geo::latlng from_;
  geo::latlng to_;
  fixed fixed_;
  capacities cap_;

  std::vector<nigiri::routing::start> from_rides_{};
  std::vector<nigiri::routing::start> to_rides_{};
  std::vector<direct_ride> direct_rides_{};

  std::vector<nigiri::routing::start> prev_from_rides_{};
  std::vector<nigiri::routing::start> prev_to_rides_{};
  std::vector<direct_ride> prev_direct_rides_{};

  std::vector<nigiri::routing::journey> odm_journeys_{};
};

}  // namespace motis::odm