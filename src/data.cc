#include "motis/data.h"

#include <filesystem>
#include <future>

#include "cista/io.h"

#include "utl/read_file.h"

#include "adr/adr.h"
#include "adr/cache.h"
#include "adr/reverse.h"
#include "adr/typeahead.h"

#include "osr/lookup.h"
#include "osr/platforms.h"
#include "osr/ways.h"

#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/timetable.h"

#include "motis/config.h"
#include "motis/constants.h"
#include "motis/elevators/parse_fasta.h"
#include "motis/match_platforms.h"
#include "motis/point_rtree.h"
#include "motis/tag_lookup.h"
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
  return out << "\nt=" << d.t_.get() << "\nr=" << d.r_ << "\ntc=" << d.tc_
             << "\nw=" << d.w_ << "\npl=" << d.pl_ << "\nl=" << d.l_
             << "\ntt=" << d.tt_.get() << "\nlocation_rtee=" << d.location_rtee_
             << "\nelevator_nodes=" << d.elevator_nodes_
             << "\nmatches=" << d.matches_ << "\nrt=" << d.rt_ << "\n";
}

data::data(std::filesystem::path p) : path_{std::move(p)} {}

data::data(std::filesystem::path p, config const& c) : path_{std::move(p)} {
  rt_ = std::make_shared<rt>();

  auto const geocoder = std::async(std::launch::async, [&]() {
    if (c.geocoding_) {
      load_geocoder();
    }
    if (c.reverse_geocoding_) {
      load_reverse_geocoder();
    }
  });

  auto const tt = std::async(std::launch::async, [&]() {
    if (c.timetable_) {
      load_tt();
    }
  });

  auto const street_routing = std::async(std::launch::async, [&]() {
    if (c.street_routing_) {
      load_osr();
    }
  });

  auto const matches = std::async(std::launch::async, [&]() {
    if (c.street_routing_ && c.timetable_) {
      load_matches();
    }
  });

  auto const elevators = std::async(std::launch::async, [&]() {
    tt.wait();
    street_routing.wait();
    matches.wait();
    if (c.elevators_) {
      load_elevators();
    }
  });

  auto const tiles = std::async(std::launch::async, [&]() {
    if (c.tiles_) {
      load_tiles();
    }
  });

  geocoder.wait();
  tt.wait();
  street_routing.wait();
  matches.wait();
  elevators.wait();
  tiles.wait();
}

data::~data() = default;
data::data(data&&) = default;
data& data::operator=(data&&) = default;

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
  tags_ = tag_lookup::read(path_ / "tags.bin");
  tt_ = n::timetable::read(path_ / "tt.bin");
  tt_->locations_.resolve_timezones();
  location_rtee_ = std::make_unique<point_rtree<n::location_idx_t>>(
      create_location_rtree(*tt_));

  auto const today = std::chrono::time_point_cast<date::days>(
      std::chrono::system_clock::now());
  rt_->rtt_ = std::make_unique<n::rt_timetable>(
      n::rt::create_rt_timetable(*tt_, today));
}

void data::load_geocoder() {
  t_ = adr::read(path_ / "adr" / "t.bin");
  tc_ = std::make_unique<adr::cache>(t_->strings_.size(), 100U);
}

void data::load_reverse_geocoder() {
  r_ = std::make_unique<adr::reverse>(path_ / "adr",
                                      cista::mmap::protection::READ);
  r_->build_rtree(*t_);
}

void data::load_matches() {
  matches_ = cista::read<platform_matches_t>(path_ / "matches.bin");
}

void data::load_elevators() {
  rt_->e_ = std::make_unique<elevators>(*w_, *elevator_nodes_,
                                        vector_map<elevator_idx_t, elevator>{});

  auto const elevator_footpath_map =
      cista::read<elevator_footpath_map_t>(path_ / "elevator_footpath_map.bin");
  update_rtt_td_footpaths(*w_, *l_, *pl_, *tt_, *location_rtee_, *rt_->e_,
                          *elevator_footpath_map, *matches_, *rt_->rtt_,
                          std::chrono::seconds{kMaxDuration});
}

void data::load_tiles() {
  auto const db_size =
      config::read(path_ / "config.yml").tiles_.value().db_size_;
  tiles_ = std::make_unique<tiles_data>(
      (path_ / "tiles" / "tiles.mdb").generic_string(), db_size);
}

}  // namespace motis