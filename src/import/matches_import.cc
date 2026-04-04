#include "motis/import/matches_import.h"

#include "cista/hashing.h"
#include "cista/io.h"
#include "cista/memory_holder.h"

#include "utl/progress_tracker.h"

#include "motis/data.h"
#include "motis/match_platforms.h"

namespace motis {

namespace fs = std::filesystem;

meta_entry_t get_way_matches_hash(config const& c) {
  return {"way_matches",
          cista::build_hash(
              c.timetable_.value_or(config::timetable{})
                  .preprocess_max_matching_distance_)};
}

matches_import::matches_import(fs::path const& data_path,
                               config const& c,
                               dataset_hashes const& h)
    : task{"matches",
           data_path,
           c,
           {h.tt_,
            h.osm_,
            osr_version(),
            n_version(),
            matches_version(),
            get_way_matches_hash(c)}} {}

matches_import::~matches_import() = default;

void matches_import::run() {
  auto d = data{data_path_, false};
  d.load_osr();
  d.load_tt("tt.bin");

  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Prepare Platform Matches").out_bounds(0, 30);

  auto const matches = platform_matches_t{get_matches(*d.tt_, *d.pl_, *d.w_)};
  cista::write(data_path_ / "matches.bin", matches);

  if (c_.timetable_->preprocess_max_matching_distance_ > 0.0) {
    progress_tracker->status("Prepare Platform Way Matches").out_bounds(30,
                                                                        100);
    auto way_matches = std::make_unique<way_matches_storage>(
        data_path_, cista::mmap::protection::WRITE,
        c_.timetable_->preprocess_max_matching_distance_);
    way_matches->preprocess_osr_matches(*d.tt_, *d.pl_, *d.w_, *d.l_,
                                        matches);
  }
}

bool matches_import::is_enabled() const {
  return c_.timetable_ && c_.use_street_routing();
}

}  // namespace motis
