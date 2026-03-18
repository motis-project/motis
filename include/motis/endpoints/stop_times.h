#pragma once

#include <vector>

#include "nigiri/rt/rt_timetable.h"
#include "nigiri/rt/run.h"
#include "nigiri/types.h"

#include "osr/location.h"
#include "osr/types.h"

#include "motis-api/motis-api.h"
#include "motis/fwd.h"
#include "motis/match_platforms.h"
#include "motis/point_rtree.h"
#include "motis/types.h"

namespace motis::ep {

struct stop_times {
  api::stoptimes_response operator()(boost::urls::url_view const&) const;

  std::vector<nigiri::rt::run> get_runs(api::stoptimes_params const&) const;
  std::vector<nigiri::rt::run> get_runs(
      api::stoptimes_params const&,
      nigiri::rt_timetable const*,
      nigiri::event_type const&,
      std::optional<nigiri::location_idx_t> const& query_stop,
      std::optional<osr::location> const& query_center) const;

  config const& config_;
  osr::ways const* w_;
  osr::platforms const* pl_;
  platform_matches_t const* matches_;
  adr_ext const* ae_;
  tz_map_t const* tz_;
  point_rtree<nigiri::location_idx_t> const& loc_rtree_;
  nigiri::timetable const& tt_;
  tag_lookup const& tags_;
  std::shared_ptr<rt> const& rt_;
};

}  // namespace motis::ep