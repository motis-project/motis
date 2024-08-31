#include "icc/data.h"

#include <filesystem>

#include "osr/lookup.h"
#include "osr/platforms.h"
#include "osr/ways.h"

#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/timetable.h"

#include "icc/match_platforms.h"
#include "icc/point_rtree.h"
#include "icc/tt_location_rtree.h"
#include "icc/update_rtt_td_footpaths.h"

namespace fs = std::filesystem;
namespace n = nigiri;

namespace icc {

void data::load(std::filesystem::path const& p, data& d) {
  d.rt_ = std::make_shared<rt>();

  if (fs::is_regular_file(p / "tt.bin")) {
    d.tt_ = n::timetable::read(cista::memory_holder{
        cista::file{(p / "tt.bin").generic_string().c_str(), "r"}.content()});
    d.tt_->locations_.resolve_timezones();
    d.location_rtee_ = std::make_unique<point_rtree<n::location_idx_t>>(
        create_location_rtree(*d.tt()));

    auto const today = std::chrono::time_point_cast<date::days>(
        std::chrono::system_clock::now());
    d.rt_->rtt_ = std::make_unique<n::rt_timetable>(
        n::rt::create_rt_timetable(*d.tt_, today));
  }

  if (fs::is_directory(p / "osr")) {
    d.w_ =
        std::make_unique<osr::ways>(p / "osr", cista::mmap::protection::READ);
    d.l_ = std::make_unique<osr::lookup>(*d.w_);
    d.elevator_nodes_ = get_elevator_nodes(*d.w_);

    if (fs::is_regular_file(p / "osr" / "node_pos.bin")) {
      d.pl_ = std::make_unique<osr::platforms>(p / "osr",
                                               cista::mmap::protection::READ);
    }
  }

  if (d.has_tt() || d.has_osr()) {
    d.rt_ = std::make_shared<rt>();
  }

  // TODO(felix) init rt->e
  /*
   auto const fasta = utl::read_file(fasta_path.generic_string().c_str());
if (!fasta.has_value()) {
fmt::println("could not read fasta file {}", fasta_path);
return 1;
}
auto const elevator_nodes = get_elevator_nodes(w);
auto e = std::make_shared<elevators>(w, elevator_nodes,
                                  parse_fasta(std::string_view{*fasta}));
   */

  if (d.has_tt() && d.has_osr()) {
    d.matches_ = std::make_unique<platform_matches_t>(
        get_matches(*d.tt(), *d.pl_, *d.w_));
    auto const elevator_footpath_map =
        read_elevator_footpath_map(p / "elevator_footpath_map.bin");
    icc::update_rtt_td_footpaths(
        *d.w_, *d.l_, *d.pl_, *d.tt(), *d.location_rtee_, *d.rt_->e_,
        *elevator_footpath_map, *d.matches_, *d.rt_->rtt_);
  }
}

data::~data() = default;

}  // namespace icc