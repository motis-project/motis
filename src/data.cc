#include "motis/data.h"

#include <filesystem>

#include "utl/read_file.h"

#include "adr/adr.h"
#include "adr/area_database.h"
#include "adr/cache.h"
#include "adr/cista_read.h"
#include "adr/reverse.h"
#include "adr/typeahead.h"

#include "osr/lookup.h"
#include "osr/platforms.h"
#include "osr/ways.h"

#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/timetable.h"

#include "motis/config.h"
#include "motis/elevators/parse_fasta.h"
#include "motis/match_platforms.h"
#include "motis/point_rtree.h"
#include "motis/tiles_data.h"
#include "motis/tt_location_rtree.h"
#include "motis/update_rtt_td_footpaths.h"

namespace fs = std::filesystem;
namespace n = nigiri;

namespace motis {

rt::rt() = default;

rt::rt(ptr<nigiri::rt_timetable>&& rtt, ptr<elevators>&& e)
    : rtt_{std::move(rtt)}, e_{std::move(e)} {}

rt::~rt() = default;

std::ostream& operator<<(std::ostream& out, data const& d) {
  return out << "\nt=" << d.t_.get() << "\narea_db=" << d.area_db_
             << "\nr=" << d.r_ << "\ntc=" << d.tc_ << "\nw=" << d.w_
             << "\npl=" << d.pl_ << "\nl=" << d.l_ << "\ntt=" << d.tt_.get()
             << "\nlocation_rtee=" << d.location_rtee_
             << "\nelevator_nodes=" << d.elevator_nodes_
             << "\nmatches=" << d.matches_ << "\nrt=" << d.rt_ << "\n";
}

data::data(std::filesystem::path p) : path_{std::move(p)} {}

data::data(std::filesystem::path p, config const& c) : path_{std::move(p)} {
  rt_ = std::make_shared<rt>();

  if (c.has_feature(feature::GEOCODING)) {
    load_geocoder();
  }

  if (c.has_feature(feature::REVERSE_GEOCODING)) {
    load_reverse_geocoder();
  }

  if (c.has_feature(feature::TIMETABLE)) {
    load_tt();
  }

  if (c.has_feature(feature::STREET_ROUTING)) {
    load_osr();
  }

  if (c.has_feature(feature::STREET_ROUTING) &&
      c.has_feature(feature::TIMETABLE)) {
    load_matches();
  }

  if (c.has_feature(feature::ELEVATORS)) {
    load_elevators();
  }

  if (c.has_feature(feature::TILES)) {
    load_tiles();
  }
}

data::~data() = default;

void data::load_osr() {
  auto const osr_path = path_ / "osr";
  w_ = std::make_unique<osr::ways>(osr_path, cista::mmap::protection::READ);
  l_ = std::make_unique<osr::lookup>(*w_);
  elevator_nodes_ =
      std::make_unique<hash_set<osr::node_idx_t>>(get_elevator_nodes(*w_));
  pl_ =
      std::make_unique<osr::platforms>(osr_path, cista::mmap::protection::READ);
  pl_->build_rtree(*w_);
}

void data::load_tt() {
  tt_ = n::timetable::read(cista::memory_holder{
      cista::file{(path_ / "tt.bin").generic_string().c_str(), "r"}.content()});
  tt_->locations_.resolve_timezones();
  location_rtee_ = std::make_unique<point_rtree<n::location_idx_t>>(
      create_location_rtree(*tt_));

  auto const today = std::chrono::time_point_cast<date::days>(
      std::chrono::system_clock::now());
  rt_->rtt_ = std::make_unique<n::rt_timetable>(
      n::rt::create_rt_timetable(*tt_, today));
}

void data::load_geocoder() {
  t_ = adr::read(path_ / "adr" / "t.bin", false);
  tc_ = std::make_unique<adr::cache>(t_->strings_.size(), 100U);
  area_db_ = std::make_unique<adr::area_database>(
      path_ / "adr", cista::mmap::protection::READ);
}

void data::load_reverse_geocoder() {
  r_ = std::make_unique<adr::reverse>(path_ / "adr",
                                      cista::mmap::protection::READ);
  r_->build_rtree(*t_);
}

void data::load_matches() {
  matches_ = std::make_unique<platform_matches_t>(get_matches(*tt_, *pl_, *w_));
}

void data::load_elevators() {
  auto const fasta =
      utl::read_file((path_ / "fasta.json").generic_string().c_str());
  utl::verify(fasta.has_value(), "could not read fasta.json");

  rt_->e_ = std::make_unique<elevators>(*w_, *elevator_nodes_,
                                        parse_fasta(std::string_view{*fasta}));

  auto const elevator_footpath_map =
      read_elevator_footpath_map(path_ / "elevator_footpath_map.bin");
  motis::update_rtt_td_footpaths(*w_, *l_, *pl_, *tt_, *location_rtee_,
                                 *rt_->e_, *elevator_footpath_map, *matches_,
                                 *rt_->rtt_);
}

void data::load_tiles() {
  auto const db_size =
      config::read(path_ / "config.yml").tiles_.value().db_size_;
  tiles_ = std::make_unique<tiles_data>(
      (path_ / "tiles" / "tiles.mdb").generic_string(), db_size);
}

}  // namespace motis