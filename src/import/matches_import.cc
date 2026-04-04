#include "motis/import/matches_import.h"

#include "cista/io.h"
#include "cista/memory_holder.h"

#include "utl/progress_tracker.h"

#include "motis/data.h"
#include "motis/match_platforms.h"

namespace motis {

namespace fs = std::filesystem;

matches_import::matches_import(fs::path const& data_path,
                               config const& c,
                               dataset_hashes const& h)
    : task{"matches",
           data_path,
           c,
           {h.tt_, h.osm_, osr_version(), n_version(), matches_version()}} {}

matches_import::~matches_import() = default;

void matches_import::run() {
  auto d = data{data_path_, false};
  d.load_osr();
  d.load_tt(c_.osr_footpath_ ? "tt_ext.bin" : "tt.bin");

  utl::get_active_progress_tracker()
      ->status("Prepare Platform Matches")
      .out_bounds(0, 100);

  auto const matches = platform_matches_t{get_matches(*d.tt_, *d.pl_, *d.w_)};
  cista::write(data_path_ / "matches.bin", matches);
}

bool matches_import::is_enabled() const {
  return c_.timetable_ && c_.use_street_routing();
}

}  // namespace motis
