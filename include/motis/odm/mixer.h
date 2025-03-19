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
  double cost(nigiri::routing::journey const& j) const;
  bool cost_dominates(nigiri::routing::journey const&,
                      nigiri::routing::journey const&) const;
  void cost_dominance(
      nigiri::pareto_set<nigiri::routing::journey> const& pt_journey,
      std::vector<nigiri::routing::journey>& odm_journeys) const;
  void productivity_dominance(
      std::vector<nigiri::routing::journey>& odm_journeys) const;
  static void pareto_dominance(
      std::vector<nigiri::routing::journey>& odm_journeys);
  void reduce_odm(std::vector<nigiri::routing::journey>& odm_journeys) const;

  double alpha_;
  double direct_taxi_penalty_;
  std::int32_t max_distance_;
  std::vector<cost_threshold> walk_cost_;
  std::vector<cost_threshold> taxi_cost_;
  std::vector<cost_threshold> transfer_cost_;
};

mixer get_default_mixer();

}  // namespace motis::odm