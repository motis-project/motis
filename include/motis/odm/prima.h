#pragma once

#include <chrono>
#include <vector>

#include "geo/latlng.h"

#include "nigiri/routing/journey.h"
#include "nigiri/routing/start_times.h"

#include "motis-api/motis-api.h"

#include "motis/fwd.h"
#include "motis/place.h"

namespace motis::ep {
struct routing;
}  // namespace motis::ep

namespace motis::odm {

constexpr auto kODMDirectPeriod = std::chrono::seconds{300};
constexpr auto kODMDirectFactor = 1.0;
constexpr auto kODMOffsetMinImprovement = std::chrono::seconds{60};
constexpr auto kODMMaxDuration = std::chrono::seconds{3600};
constexpr auto kBlacklistPath = "/api/blacklist";
constexpr auto kWhitelistPath = "/api/whitelist";
constexpr auto kRidesharingPath = "/api/whitelistRideShare";
constexpr auto kInfeasible = std::numeric_limits<nigiri::unixtime_t>::min();
static auto const kReqHeaders = std::map<std::string, std::string>{
    {"Content-Type", "application/json"}, {"Accept", "application/json"}};

using service_times_t = std::vector<nigiri::interval<nigiri::unixtime_t>>;

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

void tag_invoke(boost::json::value_from_tag const&,
                boost::json::value&,
                capacities const&);

struct prima {

  prima(std::string const& prima_url,
        osr::location const& from,
        osr::location const& to,
        api::plan_params const& query);

  void init(nigiri::interval<nigiri::unixtime_t> const& search_intvl,
            nigiri::interval<nigiri::unixtime_t> const& taxi_intvl,
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
            api::Place const& from,
            api::Place const& to,
            api::plan_params const& query,
            nigiri::routing::query const& n_query,
            unsigned api_version);

  std::size_t n_ride_sharing_events() const;

  std::string make_blacklist_taxi_request(
      nigiri::timetable const&,
      nigiri::interval<nigiri::unixtime_t> const&) const;
  bool consume_blacklist_taxi_response(std::string_view json);
  bool blacklist_taxi(nigiri::timetable const&,
                      nigiri::interval<nigiri::unixtime_t> const&);

  std::string make_whitelist_taxi_request(
      std::vector<nigiri::routing::start> const& first_mile,
      std::vector<nigiri::routing::start> const& last_mile,
      nigiri::timetable const&) const;
  bool consume_whitelist_taxi_response(
      std::string_view json,
      std::vector<nigiri::routing::journey>&,
      std::vector<nigiri::routing::start>& first_mile_taxi_rides,
      std::vector<nigiri::routing::start>& last_mile_taxi_rides);
  bool whitelist_taxi(std::vector<nigiri::routing::journey>&,
                      nigiri::timetable const&);

  std::string make_ride_sharing_request(nigiri::timetable const&) const;
  bool consume_ride_sharing_response(std::string_view json);
  bool whitelist_ride_sharing(nigiri::timetable const&);

  void extract_taxis_for_persisting(
      std::vector<nigiri::routing::journey> const& journeys);

  api::plan_params const& query_;

  boost::urls::url taxi_blacklist_;
  boost::urls::url taxi_whitelist_;
  boost::urls::url ride_sharing_whitelist_;

  osr::location const from_;
  osr::location const to_;
  nigiri::event_type fixed_;
  capacities cap_;

  std::optional<std::chrono::seconds> direct_duration_;

  std::vector<nigiri::routing::offset> first_mile_taxi_{};
  std::vector<nigiri::routing::offset> last_mile_taxi_{};
  std::vector<service_times_t> first_mile_taxi_times_{};
  std::vector<service_times_t> last_mile_taxi_times_{};
  std::vector<direct_ride> direct_taxi_{};

  std::vector<nigiri::routing::start> first_mile_ride_sharing_{};
  nigiri::vecvec<size_t, char> first_mile_ride_sharing_tour_ids_{};
  std::vector<nigiri::routing::start> last_mile_ride_sharing_{};
  nigiri::vecvec<size_t, char> last_mile_ride_sharing_tour_ids_{};
  std::vector<direct_ride> direct_ride_sharing_{};
  nigiri::vecvec<size_t, char> direct_ride_sharing_tour_ids_{};

  std::vector<nigiri::location_idx_t> whitelist_first_mile_locations_;
  std::vector<nigiri::location_idx_t> whitelist_last_mile_locations_;

  boost::json::object whitelist_response_;
};

void extract_taxis(std::vector<nigiri::routing::journey> const&,
                   std::vector<nigiri::routing::start>& first_mile_taxi_rides,
                   std::vector<nigiri::routing::start>& last_mile_taxi_rides);

void fix_first_mile_duration(
    std::vector<nigiri::routing::journey>& journeys,
    std::vector<nigiri::routing::start> const& first_mile,
    std::vector<nigiri::routing::start> const& prev_first_mile,
    nigiri::transport_mode_id_t mode);

void fix_last_mile_duration(
    std::vector<nigiri::routing::journey>& journeys,
    std::vector<nigiri::routing::start> const& last_mile,
    std::vector<nigiri::routing::start> const& prev_last_mile,
    nigiri::transport_mode_id_t mode);

std::int64_t to_millis(nigiri::unixtime_t);

nigiri::unixtime_t to_unix(std::int64_t);

std::size_t n_rides_in_response(boost::json::array const&);

std::string make_whitelist_request(
    osr::location const& from,
    osr::location const& to,
    std::vector<nigiri::routing::start> const& first_mile,
    std::vector<nigiri::routing::start> const& last_mile,
    std::vector<direct_ride> const& direct,
    nigiri::event_type fixed,
    capacities const&,
    nigiri::timetable const&);

void add_direct_odm(std::vector<direct_ride> const&,
                    std::vector<nigiri::routing::journey>&,
                    place_t const& from,
                    place_t const& to,
                    bool arrive_by,
                    nigiri::transport_mode_id_t);

}  // namespace motis::odm