#pragma once

#include "nigiri/routing/journey.h"
#include "nigiri/routing/pareto_set.h"
#include "nigiri/types.h"

#include "motis-api/motis-api.h"

namespace motis::odm {

struct cost_threshold {
  std::int32_t threshold_;
  std::int32_t cost_;
};

std::int32_t tally(std::int32_t, std::vector<cost_threshold> const&);

struct mixer {
  void mix(nigiri::pareto_set<nigiri::routing::journey> const& pt_journeys,
           std::vector<nigiri::routing::journey>& odm_journeys) const;
  std::int32_t transfer_cost(nigiri::routing::journey const&) const;
  void cost_domination(
      nigiri::pareto_set<nigiri::routing::journey> const& pt_journeys,
      std::vector<nigiri::routing::journey>& odm_journeys) const;
  void productivity_domination(
      std::vector<nigiri::routing::journey>& odm_journeys) const;

  std::vector<cost_threshold> walk_cost_;
  std::vector<cost_threshold> taxi_cost_;
  std::vector<cost_threshold> transfer_cost_;

  double direct_taxi_factor_;
  double direct_taxi_constant_;

  double travel_time_weight_;
  double distance_weight_;
  double distance_exponent_;
};

}  // namespace motis::odm