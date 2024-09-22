#include "motis/data.h"

#include <filesystem>

#include "utl/read_file.h"

#include "adr/adr.h"
#include "adr/cache.h"
#include "adr/cista_read.h"
#include "adr/reverse.h"
#include "adr/typeahead.h"

#include "osr/lookup.h"
#include "osr/platforms.h"
#include "osr/ways.h"

#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/timetable.h"

#include "motis/elevators/parse_fasta.h"
#include "motis/match_platforms.h"
#include "motis/point_rtree.h"
#include "motis/tt_location_rtree.h"
#include "motis/update_rtt_td_footpaths.h"

namespace fs = std::filesystem;
namespace n = nigiri;

namespace motis {

data::data() {}
data::~data() = default;

void data::load(std::filesystem::path const& p, data& d) {
  d.rt_ = std::make_shared<rt>();

  auto const t_path = p / "adr" / "t.bin";
  if (fs::is_regular_file(t_path)) {
    d.t_ = adr::read(t_path, false);
    d.tc_ = std::make_unique<adr::cache>(d.t_->strings_.size(), 100U);
  } else {
    fmt::println("{} not found -> not loading geo coder", t_path);
  }

  auto const r_path = p / "adr" / "r.bin";
  if (fs::is_regular_file(r_path)) {
    d.r_ = adr::cista_read<adr::reverse>(r_path, true);
  } else {
    fmt::println("{} not found -> not loading reverse geo coder", r_path);
  }

  auto const tt_path = p / "tt.bin";
  if (fs::is_regular_file(tt_path)) {
    d.tt_ = n::timetable::read(cista::memory_holder{
        cista::file{(tt_path).generic_string().c_str(), "r"}.content()});
    d.tt_->locations_.resolve_timezones();
    d.location_rtee_ = std::make_unique<point_rtree<n::location_idx_t>>(
        create_location_rtree(*d.tt()));

    auto const today = std::chrono::time_point_cast<date::days>(
        std::chrono::system_clock::now());
    d.rt_->rtt_ = std::make_unique<n::rt_timetable>(
        n::rt::create_rt_timetable(*d.tt_, today));
  } else {
    fmt::println("{} not found -> not loading timetable", tt_path);
  }

  auto const osr_path = p / "osr";
  if (fs::is_directory(osr_path)) {
    d.w_ =
        std::make_unique<osr::ways>(p / "osr", cista::mmap::protection::READ);
    d.l_ = std::make_unique<osr::lookup>(*d.w_);
    d.elevator_nodes_ =
        std::make_unique<hash_set<osr::node_idx_t>>(get_elevator_nodes(*d.w_));

    if (fs::is_regular_file(osr_path / "node_pos.bin")) {
      d.pl_ = std::make_unique<osr::platforms>(p / "osr",
                                               cista::mmap::protection::READ);
      d.pl_->build_rtree(*d.w_);
    }
  } else {
    fmt::println("{} not found -> not loading street routing", osr_path);
  }

  auto const fasta_path = p / "fasta.json";
  if (fs::is_regular_file(fasta_path)) {
    auto const fasta = utl::read_file((fasta_path).generic_string().c_str());
    utl::verify(fasta.has_value(), "could not read fasta.json");

    d.rt_->e_ = std::make_unique<elevators>(
        *d.w_, *d.elevator_nodes_, parse_fasta(std::string_view{*fasta}));
  } else {
    fmt::println("{} not found -> not loading elevator status", fasta_path);
  }

  if (d.has_tt() && d.has_osr() && d.pl_ != nullptr && d.rt_->e_ != nullptr &&
      d.rt_->rtt_ != nullptr) {
    d.matches_ = std::make_unique<platform_matches_t>(
        get_matches(*d.tt(), *d.pl_, *d.w_));
    auto const elevator_footpath_map =
        read_elevator_footpath_map(p / "elevator_footpath_map.bin");
    motis::update_rtt_td_footpaths(
        *d.w_, *d.l_, *d.pl_, *d.tt(), *d.location_rtee_, *d.rt_->e_,
        *elevator_footpath_map, *d.matches_, *d.rt_->rtt_);
  } else {
    fmt::println("not updating footpaths according to elevator status");
  }
}

}  // namespace motis