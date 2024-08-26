#include "icc/data.h"

#include <filesystem>

#include "osr/lookup.h"
#include "osr/platforms.h"
#include "osr/ways.h"

#include "nigiri/timetable.h"

#include "icc/match_platforms.h"
#include "icc/point_rtree.h"
#include "icc/tt_location_rtree.h"
#include "icc/update_rtt_td_footpaths.h"

namespace fs = std::filesystem;
namespace n = nigiri;

namespace icc {

data::data(std::filesystem::path const& p) {
  if (fs::is_regular_file(p / "tt.bin")) {
    tt_ = std::make_unique<cista::wrapped<n::timetable>>(
        n::timetable::read(cista::memory_holder{
            cista::file{(p / "tt.bin").generic_string().c_str(), "r"}
                .content()}));
    (*tt_)->locations_.resolve_timezones();
    location_rtee_ = std::make_unique<point_rtree<n::location_idx_t>>(
        create_location_rtree(tt()));
  }

  if (fs::is_directory(p / "osr")) {
    w_ = std::make_unique<osr::ways>(p / "osr", cista::mmap::protection::READ);
    l_ = std::make_unique<osr::lookup>(*w_);

    if (fs::is_regular_file(p / "osr" / "node_pos.bin")) {
      pl_ = std::make_unique<osr::platforms>(p / "osr",
                                             cista::mmap::protection::READ);
    }
  }

  if (has_tt() || has_osr()) {
    rt_ = std::make_shared<rt>();
  }

  if (has_tt() && has_osr()) {
    matches_ =
        std::make_unique<platform_matches_t>(get_matches(tt(), *pl_, *w_));
    auto const elevator_footpath_map =
        read_elevator_footpath_map(p / "elevator_footpath_map.bin");
    icc::update_rtt_td_footpaths(*w_, *l_, *pl_, tt(), *location_rtee_,
                                 *elevator_footpath_map, matches_, *rt_);
  }
}

data::~data() = default;

}  // namespace icc