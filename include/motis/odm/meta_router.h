#pragma once

#include <optional>
#include <variant>
#include <vector>

#include "osr/location.h"

#include "nigiri/types.h"

#include "motis-api/motis-api.h"

#include "motis/endpoints/routing.h"
#include "motis/fwd.h"
#include "motis/gbfs/routing_data.h"
#include "motis/odm/query_factory.h"
#include "motis/place.h"

namespace nigiri {
struct timetable;
struct rt_timetable;
}  // namespace nigiri

namespace nigiri::routing {
struct query;
struct journey;
}  // namespace nigiri::routing

namespace motis::odm {

struct prima;

struct meta_router {
  meta_router(ep::routing const&,
              api::plan_params const&,
              std::vector<api::ModeEnum> const& pre_transit_modes,
              std::vector<api::ModeEnum> const& post_transit_modes,
              std::vector<api::ModeEnum> const& direct_modes,
              std::variant<osr::location, tt_location> const& from,
              std::variant<osr::location, tt_location> const& to,
              api::Place const& from_p,
              api::Place const& to_p,
              nigiri::routing::query const& start_time,
              std::vector<api::Itinerary>& direct,
              nigiri::duration_t fastest_direct_,
              bool odm_pre_transit,
              bool odm_post_transit,
              bool odm_direct,
              unsigned api_version);
  ~meta_router();

  api::plan_response run();

  struct routing_result {
    routing_result() = default;
    routing_result(
        nigiri::routing::routing_result<nigiri::routing::raptor_stats> rr)
        : journeys_{*rr.journeys_},
          interval_{rr.interval_},
          search_stats_{rr.search_stats_},
          algo_stats_{rr.algo_stats_} {}

    nigiri::pareto_set<nigiri::routing::journey> journeys_{};
    nigiri::interval<nigiri::unixtime_t> interval_{};
    nigiri::routing::search_stats search_stats_{};
    nigiri::routing::raptor_stats algo_stats_{};
  };

private:
  void init_prima(nigiri::interval<nigiri::unixtime_t> const& search_intvl,
                  nigiri::interval<nigiri::unixtime_t> const& odm_intvl);
  nigiri::routing::query get_base_query(
      nigiri::interval<nigiri::unixtime_t> const&) const;
  std::vector<routing_result> search_interval(
      std::vector<nigiri::routing::query> const&) const;
  void add_direct() const;

  ep::routing const& r_;
  api::plan_params const& query_;
  std::vector<api::ModeEnum> const& pre_transit_modes_;
  std::vector<api::ModeEnum> const& post_transit_modes_;
  std::vector<api::ModeEnum> const& direct_modes_;
  std::variant<osr::location, tt_location> const& from_;
  std::variant<osr::location, tt_location> const& to_;
  api::Place const& from_place_;
  api::Place const& to_place_;
  nigiri::routing::query const& start_time_;
  std::vector<api::Itinerary>& direct_;
  nigiri::duration_t fastest_direct_;
  bool odm_pre_transit_;
  bool odm_post_transit_;
  bool odm_direct_;
  unsigned api_version_;

  nigiri::timetable const* tt_;
  std::shared_ptr<rt> const rt_;
  nigiri::rt_timetable const* rtt_;
  motis::elevators const* e_;
  gbfs::gbfs_routing_data gbfs_rd_;
  std::variant<osr::location, tt_location> const& start_;
  std::variant<osr::location, tt_location> const& dest_;
  std::vector<api::ModeEnum> start_modes_;
  std::vector<api::ModeEnum> dest_modes_;

  std::optional<std::vector<api::RentalFormFactorEnum>> const&
      start_form_factors_;
  std::optional<std::vector<api::RentalFormFactorEnum>> const&
      dest_form_factors_;
  std::optional<std::vector<api::RentalPropulsionTypeEnum>> const&
      start_propulsion_types_;
  std::optional<std::vector<api::RentalPropulsionTypeEnum>> const&
      dest_propulsion_types_;
  std::optional<std::vector<std::string>> const& start_rental_providers_;
  std::optional<std::vector<std::string>> const& dest_rental_providers_;
  bool start_ignore_rental_return_constraints_{};
  bool dest_ignore_rental_return_constraints_{};

  std::unique_ptr<prima> p_;
};

}  // namespace motis::odm
