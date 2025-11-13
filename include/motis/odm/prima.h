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

  std::size_t n_taxi_events() const;
  std::size_t n_ride_sharing_events() const;

  std::string make_taxi_request(nigiri::timetable const&) const;

  bool consume_blacklist_taxis_response(std::string_view json);
  bool blacklist_taxis(nigiri::timetable const&);

  void extract_taxis(std::vector<nigiri::routing::journey> const&);
  bool consume_whitelist_taxis_response(std::string_view json,
                                        std::vector<nigiri::routing::journey>&);
  bool whitelist_taxis(std::vector<nigiri::routing::journey>&,
                       nigiri::timetable const&);

  void add_direct_odm(std::vector<direct_ride> const&,
                      std::vector<nigiri::routing::journey>&,
                      place_t const& from,
                      place_t const& to,
                      bool arrive_by,
                      nigiri::transport_mode_id_t) const;

  std::string make_ride_sharing_request(nigiri::timetable const&) const;

  bool consume_whitelist_ride_sharing_response(std::string_view json);
  bool whitelist_ride_sharing(nigiri::timetable const&);

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

  boost::urls::url taxi_blacklist_;
  boost::urls::url taxi_whitelist_;
  boost::urls::url ride_sharing_whitelist_;

  osr::location const from_;
  osr::location const to_;
  nigiri::event_type fixed_;
  capacities cap_;

  std::vector<nigiri::routing::start> first_mile_taxi_{};
  std::vector<nigiri::routing::start> last_mile_taxi_{};
  std::vector<direct_ride> direct_taxi_{};

  std::vector<nigiri::routing::start> first_mile_ride_sharing_{};
  nigiri::vecvec<size_t, char> first_mile_ride_sharing_tour_ids_{};
  std::vector<nigiri::routing::start> last_mile_ride_sharing_{};
  nigiri::vecvec<size_t, char> last_mile_ride_sharing_tour_ids_{};
  std::vector<direct_ride> direct_ride_sharing_{};
  nigiri::vecvec<size_t, char> direct_ride_sharing_tour_ids_{};
};

}  // namespace motis::odm