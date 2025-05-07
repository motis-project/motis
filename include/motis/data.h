#pragma once

#include <memory>

#include "prometheus/registry.h"

#include "cista/memory_holder.h"

#include "date/date.h"

#include "nigiri/types.h"

#include "osr/types.h"

#include "motis/compute_footpaths.h"
#include "motis/config.h"
#include "motis/fwd.h"
#include "motis/gbfs/data.h"
#include "motis/match_platforms.h"
#include "motis/types.h"

namespace motis {

struct elevators;

template <typename T>
struct point_rtree;

template <typename T>
using ptr = std::unique_ptr<T>;

struct rt {
  rt();
  rt(ptr<nigiri::rt_timetable>&&, ptr<elevators>&&, ptr<railviz_rt_index>&&);
  ~rt();
  ptr<nigiri::rt_timetable> rtt_;
  ptr<railviz_rt_index> railviz_rt_;
  ptr<elevators> e_;
};

struct data {
  data(std::filesystem::path);
  data(std::filesystem::path, config const&);
  ~data();

  data(data const&) = delete;
  data& operator=(data const&) = delete;

  data(data&&);
  data& operator=(data&&);

  friend std::ostream& operator<<(std::ostream&, data const&);

  void load_osr();
  void load_tt(std::filesystem::path const&);
  void load_shapes();
  void load_railviz();
  void load_geocoder();
  void load_matches();
  void load_reverse_geocoder();
  void load_tiles();

  void init_rtt(date::sys_days = std::chrono::time_point_cast<date::days>(
                    std::chrono::system_clock::now()));

  auto cista_members() {
    // !!! Remember to add all new members !!!
    return std::tie(config_, t_, r_, tc_, w_, pl_, l_, elevations_, tt_, tags_,
                    location_rtree_, elevator_nodes_, shapes_, railviz_static_,
                    matches_, rt_, gbfs_, odm_bounds_, metrics_);
  }

  std::filesystem::path path_;
  config config_;

  cista::wrapped<adr::typeahead> t_;
  ptr<adr::reverse> r_;
  ptr<adr::cache> tc_;
  ptr<osr::ways> w_;
  ptr<osr::platforms> pl_;
  ptr<osr::lookup> l_;
  ptr<osr::elevation_storage> elevations_;
  cista::wrapped<nigiri::timetable> tt_;
  cista::wrapped<tag_lookup> tags_;
  ptr<point_rtree<nigiri::location_idx_t>> location_rtree_;
  ptr<hash_set<osr::node_idx_t>> elevator_nodes_;
  ptr<nigiri::shapes_storage> shapes_;
  ptr<railviz_static_index> railviz_static_;
  cista::wrapped<platform_matches_t> matches_;
  ptr<tiles_data> tiles_;
  std::shared_ptr<rt> rt_{std::make_shared<rt>()};
  std::shared_ptr<gbfs::gbfs_data> gbfs_{};
  ptr<odm::bounds> odm_bounds_;
  ptr<prometheus::Registry> metrics_{std::make_unique<prometheus::Registry>()};
};

}  // namespace motis
