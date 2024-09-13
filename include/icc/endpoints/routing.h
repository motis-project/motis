#pragma once

#include "boost/json/value.hpp"

#include "nigiri/routing/query.h"

#include "osr/location.h"
#include "osr/types.h"

#include "icc-api/icc-api.h"
#include "icc/elevators/elevators.h"
#include "icc/fwd.h"

namespace icc::ep {

struct routing {
  api::plan_response operator()(boost::urls::url_view const& url) const;

  std::vector<nigiri::routing::offset> get_offsets(
      osr::location const&,
      osr::direction,
      std::vector<api::ModeEnum> const&,
      bool wheelchair,
      std::chrono::seconds max) const;

  nigiri::hash_map<nigiri::location_idx_t,
                   std::vector<nigiri::routing::td_offset>>
  get_td_offsets(elevators const&,
                 osr::location const&,
                 osr::direction,
                 std::vector<api::ModeEnum> const&,
                 bool wheelchair,
                 std::chrono::seconds max) const;

  osr::ways const& w_;
  osr::lookup const& l_;
  osr::platforms const& pl_;
  nigiri::timetable const& tt_;
  point_rtree<nigiri::location_idx_t> const& loc_tree_;
  vector_map<nigiri::location_idx_t, osr::platform_idx_t> const& matches_;
  std::shared_ptr<rt> const& rt_;
};

}  // namespace icc::ep