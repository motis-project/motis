#pragma once

#include "nigiri/routing/journey.h"
#include "nigiri/routing/pareto_set.h"
#include "nigiri/types.h"

#include "motis-api/motis-api.h"
#include "motis/fwd.h"

namespace motis::odm {

struct cost_threshold {
  friend cost_threshold tag_invoke(boost::json::value_to_tag<cost_threshold>,
                                   boost::json::value const&);
  friend void tag_invoke(boost::json::value_from_tag,
                         boost::json::value&,
                         cost_threshold const&);

  std::int64_t threshold_;
  double cost_;
};

double tally(std::int64_t, std::vector<cost_threshold> const&);

struct mixer {
  void mix(nigiri::pareto_set<nigiri::routing::journey> const& pt_journeys,
           std::vector<nigiri::routing::journey>& odm_journeys,
           std::vector<nigiri::routing::journey> const& ride_share_journeys,
           metrics_registry* metrics,
           std::optional<std::string_view> stats_path) const;
  static void pareto_dominance(
      std::vector<nigiri::routing::journey>& odm_journeys);
  double transfer_cost(nigiri::routing::journey const&) const;
  double cost(nigiri::routing::journey const& j) const;
  std::vector<double> get_threshold(
      std::vector<nigiri::routing::journey> const&,
      nigiri::interval<nigiri::unixtime_t> const&,
      double slope) const;
  void write_journeys(
      nigiri::pareto_set<nigiri::routing::journey> const& pt_journeys,
      std::vector<nigiri::routing::journey> const& odm_journeys,
      std::string_view stats_path) const;

  friend std::ostream& operator<<(std::ostream&, mixer const&);

  friend mixer tag_invoke(boost::json::value_to_tag<mixer>,
                          boost::json::value const&);
  friend void tag_invoke(boost::json::value_from_tag,
                         boost::json::value&,
                         mixer const&);

  double direct_taxi_penalty_;
  double pt_slope_;
  double odm_slope_;
  std::vector<cost_threshold> taxi_cost_;
  std::vector<cost_threshold> transfer_cost_;
};

std::vector<nigiri::routing::journey> get_mixer_input(
    nigiri::pareto_set<nigiri::routing::journey> const& pt_journeys,
    std::vector<nigiri::routing::journey> const& odm_journeys);

mixer get_default_mixer();

}  // namespace motis::odm