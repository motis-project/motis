#pragma once

#include <memory>

#include "cista/memory_holder.h"

#include "date/date.h"

#include "nigiri/rt/vdv_aus.h"
#include "nigiri/types.h"

#include "osr/types.h"

#include "motis/adr_extend_tt.h"
#include "motis/config.h"
#include "motis/elevators/parse_elevator_id_osm_mapping.h"
#include "motis/fwd.h"
#include "motis/gbfs/data.h"
#include "motis/match_platforms.h"
#include "motis/rt/auser.h"
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
  void load_flex_areas();
  void load_shapes();
  void load_railviz();
  void load_tbd();
  void load_geocoder();
  void load_matches();
  void load_way_matches();
  void load_reverse_geocoder();
  void load_tiles();
  void load_auser_updater(std::string_view, config::timetable::dataset const&);

  void init_rtt(date::sys_days = std::chrono::time_point_cast<date::days>(
                    std::chrono::system_clock::now()));

  auto cista_members() {
    // !!! Remember to add all new members !!!
    return std::tie(
        config_, t_, adr_ext_, f_, tz_, r_, tc_, w_, pl_, l_, elevations_, tt_,
        tbd_, tags_, location_rtree_, elevator_nodes_, elevator_osm_mapping_,
        shapes_, railviz_static_, matches_, way_matches_, rt_, gbfs_,
        odm_bounds_, ride_sharing_bounds_, flex_areas_, metrics_, auser_);
  }

  std::filesystem::path path_;
  config config_;

  cista::wrapped<adr::typeahead> t_;
  cista::wrapped<adr_ext> adr_ext_;
  ptr<adr::formatter> f_;
  ptr<vector_map<adr_extra_place_idx_t, date::time_zone const*>> tz_;
  ptr<adr::reverse> r_;
  ptr<adr::cache> tc_;
  ptr<osr::ways> w_;
  ptr<osr::platforms> pl_;
  ptr<osr::lookup> l_;
  ptr<osr::elevation_storage> elevations_;
  cista::wrapped<nigiri::timetable> tt_;
  cista::wrapped<nigiri::routing::tb::tb_data> tbd_;
  cista::wrapped<tag_lookup> tags_;
  ptr<point_rtree<nigiri::location_idx_t>> location_rtree_;
  ptr<hash_set<osr::node_idx_t>> elevator_nodes_;
  ptr<elevator_id_osm_mapping_t> elevator_osm_mapping_;
  ptr<nigiri::shapes_storage> shapes_;
  ptr<railviz_static_index> railviz_static_;
  cista::wrapped<vector_map<nigiri::location_idx_t, osr::platform_idx_t>>
      matches_;
  ptr<way_matches_storage> way_matches_;
  ptr<tiles_data> tiles_;
  std::shared_ptr<rt> rt_{std::make_shared<rt>()};
  std::shared_ptr<gbfs::gbfs_data> gbfs_{};
  ptr<odm::bounds> odm_bounds_;
  ptr<odm::ride_sharing_bounds> ride_sharing_bounds_;
  ptr<flex::flex_areas> flex_areas_;
  ptr<metrics_registry> metrics_;
  ptr<std::map<std::string, auser>> auser_;
};

}  // namespace motis
