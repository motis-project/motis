#pragma once

#include <memory>

#include "cista/memory_holder.h"

#include "nigiri/types.h"

#include "osr/types.h"

#include "motis/compute_footpaths.h"
#include "motis/fwd.h"
#include "motis/match_platforms.h"
#include "motis/types.h"

namespace motis {

struct elevators;

template <typename T>
struct point_rtree;

template <typename T>
using ptr = std::unique_ptr<T>;

struct rt {
  ptr<nigiri::rt_timetable> rtt_;
  ptr<elevators> e_;
};

struct data {
  data();
  ~data();

  data(data const&) = delete;
  data(data&&) = delete;

  data& operator=(data const&) = delete;
  data& operator=(data&&) = delete;

  static void load(std::filesystem::path const&, data&);

  bool has_tt() const { return tt_.get() != nullptr; }
  bool has_osr() const { return w_ != nullptr; }
  bool has_platforms() const { return pl_ != nullptr; }

  nigiri::timetable const* tt() const { return tt_.get(); }

  auto cista_members() {
    return std::tie(t_, r_, tc_, w_, pl_, l_, tt_, location_rtee_,
                    elevator_nodes_, matches_, rt_);
  }

  cista::wrapped<adr::typeahead> t_;
  ptr<adr::reverse> r_;
  ptr<adr::cache> tc_;
  ptr<osr::ways> w_;
  ptr<osr::platforms> pl_;
  ptr<osr::lookup> l_;
  cista::wrapped<nigiri::timetable> tt_;
  ptr<point_rtree<nigiri::location_idx_t>> location_rtee_;
  ptr<hash_set<osr::node_idx_t>> elevator_nodes_;
  ptr<platform_matches_t> matches_;
  std::shared_ptr<rt> rt_;
};

}  // namespace motis