#pragma once

#include "boost/url/url_view.hpp"

#include "adr/reverse.h"
#include "adr/typeahead.h"

#include "nigiri/rt/rt_timetable.h"

// #include "motis/endpoints/routing.h"
#include "motis-api/motis-api.h"
#include "motis/data.h"
#include "motis/fwd.h"
#include "motis/point_rtree.h"

namespace motis::ep {

// struct one_to_all : public routing {
struct one_to_all {
  api::Reachable operator()(boost::urls::url_view const&) const;

  config const& config_;
  osr::ways const* w_;
  osr::lookup const* l_;
  osr::platforms const* pl_;
  nigiri::timetable const* tt_;
  tag_lookup const* tags_;
  point_rtree<nigiri::location_idx_t> const* loc_tree_;
  platform_matches_t const* matches_;
  std::shared_ptr<rt> const& rt_;
  nigiri::shapes_storage const* shapes_;
  std::shared_ptr<gbfs::gbfs_data> const& gbfs_;
  odm::bounds const* odm_bounds_;
  // osr::ways const& w_;
  // osr::lookup const& l_;
  // std::shared_ptr<gbfs::gbfs_data> const& gbfs_;
  // nigiri::timetable const* tt_;
  // // std::shared_ptr<rt> const& rt_;
  // // nigiri::rt_timetable  const* rtt_;
  // // osr::platforms const* pl_;
  // // nigiri::timetable const* tt_;
  // // tag_lookup const* tags_;
  // // point_rtree<nigiri::location_idx_t> const* loc_tree_;
  // // platform_matches_t const* matches_;
  // // std::shared_ptr<rt> const& rt_;
  // // nigiri::shapes_storage const* shapes_;

  adr::typeahead const& t_;
  adr::reverse const& r_;
};

}  // namespace motis::ep
