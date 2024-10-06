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

struct config;
struct elevators;

template <typename T>
struct point_rtree;

template <typename T>
using ptr = std::unique_ptr<T>;

struct rt {
  rt();
  rt(ptr<nigiri::rt_timetable>&&, ptr<elevators>&&);
  ~rt();
  ptr<nigiri::rt_timetable> rtt_;
  ptr<elevators> e_;
};

struct data {
  data(std::filesystem::path);
  data(std::filesystem::path, config const&);
  ~data();

  data(data const&) = delete;
  data(data&&) = delete;

  data& operator=(data const&) = delete;
  data& operator=(data&&) = delete;

  friend std::ostream& operator<<(std::ostream&, data const&);

  void load_osr();
  void load_tt();
  void load_geocoder();
  void load_matches();
  void load_reverse_geocoder();
  void load_elevators();
  void load_tiles();

  auto cista_members() {
    // !!! Remember to add all new members !!!
    return std::tie(t_, r_, tc_, w_, pl_, l_, tt_, location_rtee_,
                    elevator_nodes_, matches_, rt_);
  }

  std::filesystem::path path_;
  cista::wrapped<adr::typeahead> t_;
  ptr<adr::area_database> area_db_;
  ptr<adr::reverse> r_;
  ptr<adr::cache> tc_;
  ptr<osr::ways> w_;
  ptr<osr::platforms> pl_;
  ptr<osr::lookup> l_;
  cista::wrapped<nigiri::timetable> tt_;
  ptr<point_rtree<nigiri::location_idx_t>> location_rtee_;
  ptr<hash_set<osr::node_idx_t>> elevator_nodes_;
  ptr<platform_matches_t> matches_;
  ptr<tiles_data> tiles_;
  std::shared_ptr<rt> rt_{std::make_shared<rt>()};
};

}  // namespace motis