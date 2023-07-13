#pragma once

#include "boost/range/irange.hpp"

#include "motis/footpaths/platforms.h"
#include "motis/ppr/profile_info.h"

#include "motis/module/module.h"

namespace fs = std::filesystem;

namespace motis::footpaths {

struct footpaths : public motis::module::module {
  footpaths();
  ~footpaths() override;

  footpaths(footpaths const&) = delete;
  footpaths& operator=(footpaths const&) = delete;

  footpaths(footpaths&&) = delete;
  footpaths& operator=(footpaths&&) = delete;

  void init(motis::module::registry&) override;
  void import(motis::module::import_dispatcher&) override;
  bool import_successful() const override { return import_successful_; };

private:
  // directories
  fs::path module_data_dir() const;

  /**
   * Matches any location to one imported osm platform (or station). Updates the
   * osm_id and type of a location (when match is found)
   *
   * The match distance is iteratively increased from match_distance_min to
   * match_distance_max in step size match_distance_step_.
   *
   * Bus Stops are only matched up to a distance of match_bus_stop_distance_.
   *
   * 1st. for each location exact matches by name are searched.
   * 2nd. (if 1st failed) for each location only match platform number.
   *
   * @return number of matched platforms
   *
   */
  u_int match_locations_and_platforms();

  bool match_by_distance(
      nigiri::location_idx_t const i,
      boost::strided_integer_range<int> match_distances,
      std::function<bool(std::string, std::string_view)> matches);

  int max_walk_duration_{60};

  int match_distance_min_{0};
  int match_distance_max_{400};
  int match_distance_step_{40};
  int match_bus_stop_max_distance_{120};

  // initialize ppr routing data
  std::size_t edge_rtree_max_size_{1024UL * 1024 * 1024 * 3};
  std::size_t area_rtree_max_size_{1024UL * 1024 * 1024};
  bool lock_rtrees_{false};
  bool ppr_exact_{true};

  struct impl;
  std::unique_ptr<impl> impl_;
  std::map<std::string, ppr::profile_info> ppr_profiles_;
  std::vector<ppr::profile_info> profiles_;
  std::map<std::string, size_t> ppr_profile_pos_;
  std::unique_ptr<platforms> platforms_;
  bool import_successful_{false};
};

}  // namespace motis::footpaths
