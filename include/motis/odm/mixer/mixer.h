#pragma once

#include "boost/json/value_to.hpp"

#include "nigiri/routing/journey.h"
#include "nigiri/routing/pareto_set.h"
#include "nigiri/types.h"

#include "motis-api/motis-api.h"
#include "motis/fwd.h"

namespace motis::odm {

struct cost_threshold {
  std::int32_t threshold_;
  std::int32_t cost_;
};

cost_threshold tag_invoke(boost::json::value_to_tag<cost_threshold> const&, boost::json::value const&);

std::int32_t tally(std::int32_t, std::vector<cost_threshold> const&);

struct mixer {
  void mix(nigiri::pareto_set<nigiri::routing::journey> const& pt_journeys,
           std::vector<nigiri::routing::journey>& odm_journeys,
           metrics_registry* metrics, std::optional<std::string_view> stats_path) const;
  static void pareto_dominance(
    std::vector<nigiri::routing::journey>& odm_journeys);
  std::int32_t transfer_cost(nigiri::routing::journey const&) const;
  double cost(nigiri::routing::journey const& j) const;
  void cost_dominance(
      nigiri::pareto_set<nigiri::routing::journey> const& pt_journeys,
      std::vector<nigiri::routing::journey>& odm_journeys,
      std::optional<std::string_view> stats_path) const;

  double direct_taxi_penalty_;
  std::int32_t max_distance_;
  std::vector<cost_threshold> walk_cost_;
  std::vector<cost_threshold> taxi_cost_;
  std::vector<cost_threshold> transfer_cost_;
};

std::vector<nigiri::routing::journey> get_mixer_input(
    nigiri::pareto_set<nigiri::routing::journey> const& pt_journeys,
    std::vector<nigiri::routing::journey> const& odm_journeys);

mixer get_default_mixer();

mixer tag_invoke(boost::json::value_to_tag<mixer> const&, boost::json::value const&);

}  // namespace motis::odm