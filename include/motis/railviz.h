#pragma once

#include <filesystem>
#include <memory>

#include "geo/box.h"

#include "nigiri/types.h"

#include "motis-api/motis-api.h"
#include "motis/fwd.h"
#include "motis/match_platforms.h"

namespace motis {

struct railviz_bounding_boxes {
  railviz_bounding_boxes(std::filesystem::path const&,
                         nigiri::timetable const&,
                         nigiri::shapes_storage const*);
  railviz_bounding_boxes(std::filesystem::path const&);
  geo::box get_bounding_box(nigiri::route_idx_t const) const;
  geo::box get_bounding_box_or_else(nigiri::route_idx_t,
                                    std::size_t,
                                    std::function<geo::box()> const&) const;
  nigiri::mm_vecvec<nigiri::route_idx_t, geo::box> boxes_;
};

struct railviz_static_index {
  railviz_static_index(nigiri::timetable const&, railviz_bounding_boxes&&);
  ~railviz_static_index();

  struct impl;
  std::unique_ptr<impl> impl_;
};

struct railviz_rt_index {
  railviz_rt_index(nigiri::timetable const&, nigiri::rt_timetable const&);
  ~railviz_rt_index();

  struct impl;
  std::unique_ptr<impl> impl_;
};

api::trips_response get_trains(tag_lookup const&,
                               nigiri::timetable const&,
                               nigiri::rt_timetable const*,
                               nigiri::shapes_storage const*,
                               osr::ways const* w,
                               osr::platforms const* pl,
                               platform_matches_t const* matches,
                               railviz_static_index::impl const&,
                               railviz_rt_index::impl const&,
                               api::trips_params const&);

}  // namespace motis