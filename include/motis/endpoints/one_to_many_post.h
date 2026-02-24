#pragma once

#include <memory>

#include "nigiri/types.h"

#include "motis-api/motis-api.h"

#include "motis/data.h"
#include "motis/fwd.h"
#include "motis/match_platforms.h"
#include "motis/point_rtree.h"

namespace motis::ep {

struct one_to_many_post {
  api::oneToManyPost_response operator()(
      motis::api::OneToManyParams const&) const;

  osr::ways const& w_;
  osr::lookup const& l_;
  osr::elevation_storage const* elevations_;
};

struct one_to_many_intermodal_post {
  api::oneToManyIntermodalPost_response operator()(
      api::OneToManyIntermodalParams const&) const;

  config const& config_;
  osr::ways const* w_;
  osr::lookup const* l_;
  osr::platforms const* pl_;
  osr::elevation_storage const* elevations_;
  nigiri::timetable const& tt_;
  std::shared_ptr<rt> const& rt_;
  tag_lookup const& tags_;
  flex::flex_areas const* fa_;
  point_rtree<nigiri::location_idx_t> const* loc_tree_;
  platform_matches_t const* matches_;
  way_matches_storage const* way_matches_;
  std::shared_ptr<gbfs::gbfs_data> const& gbfs_;
  metrics_registry* metrics_;
};

}  // namespace motis::ep
