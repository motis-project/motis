#pragma once

#include <chrono>
#include <vector>

#include "geo/latlng.h"

#include "nigiri/routing/journey.h"
#include "nigiri/routing/start_times.h"

#include "motis-api/motis-api.h"

#include "motis/fwd.h"

using namespace std::chrono_literals;

constexpr auto const kODMLookAhead = nigiri::duration_t{24h};
constexpr auto const kSearchIntervalSize = nigiri::duration_t{6h};
constexpr auto const kContextPadding = nigiri::duration_t{2h};
constexpr auto const kODMDirectPeriod = 300s;
constexpr auto const kODMDirectFactor = 1.0;
constexpr auto const kODMOffsetMinImprovement = 60s;
constexpr auto const kODMMaxDuration = 3600s;
constexpr auto const kBlacklistPath = "/api/blacklist";
constexpr auto const kWhitelistPath = "/api/whitelist";
static auto const kReqHeaders = std::map<std::string, std::string>{
    {"Content-Type", "application/json"}, {"Accept", "application/json"}};

namespace motis::ep {
struct routing;
}  // namespace motis::ep

namespace motis::odm {

struct direct_ride {
  nigiri::unixtime_t dep_;
  nigiri::unixtime_t arr_;
};

struct capacities {
  std::int64_t wheelchairs_;
  std::int64_t bikes_;
  std::int64_t passengers_;
  std::int64_t luggage_;
};

struct prima {

  prima(osr::location const& from,
        osr::location const& to,
        api::plan_params const& query);

  void init(nigiri::interval<nigiri::unixtime_t> const& search_intvl,
            nigiri::interval<nigiri::unixtime_t> const& odm_intvl,
            bool use_first_mile_taxi,
            bool use_last_mile_taxi,
            bool use_direct_taxi,
            bool use_first_mile_ride_sharing,
            bool use_last_mile_ride_sharing,
            bool use_direct_ride_sharing,
            nigiri::timetable const& tt,
            nigiri::rt_timetable const* rtt,
            ep::routing const& r,
            elevators const* e,
            gbfs::gbfs_routing_data& gbfs,
            api::Place const& from_p,
            api::Place const& to_p,
            api::plan_params const& query,
            nigiri::routing::query const& n_query,
            unsigned api_version);

  std::string get_prima_request(nigiri::timetable const&) const;
  std::size_t n_events() const;
  bool blacklist_update(std::string_view json);
  bool whitelist_update(std::string_view json);
  void adjust_to_whitelisting();

  osr::location const& from_;
  osr::location const& to_;
  nigiri::event_type fixed_;
  capacities cap_;

  std::vector<nigiri::routing::start> first_mile_ride_sharing_{};
  std::vector<nigiri::routing::start> last_mile_ride_sharing_{};
  std::vector<direct_ride> direct_ride_sharing_{};

  std::vector<nigiri::routing::start> first_mile_taxi_{};
  std::vector<nigiri::routing::start> last_mile_taxi_{};
  std::vector<direct_ride> direct_taxi_{};

  std::vector<nigiri::routing::journey> odm_journeys_{};
};

}  // namespace motis::odm