#include "motis/import/way_matches_import.h"

#include "cista/hashing.h"
#include "cista/io.h"

#include "utl/progress_tracker.h"

#include "motis/data.h"
#include "motis/match_platforms.h"

namespace motis {

namespace fs = std::filesystem;

meta_entry_t get_way_matches_hash(config const& c) {
  auto const tt = c.timetable_.value_or(config::timetable{});
  return {"way_matches",
          cista::build_hash(c.osr_footpath_, tt.use_osm_stop_coordinates_,
                            tt.extend_missing_footpaths_,
                            tt.max_matching_distance_,
                            tt.max_footpath_length_,
                            tt.preprocess_max_matching_distance_)};
}

way_matches_import::way_matches_import(fs::path const& data_path,
                                       config const& c,
                                       dataset_hashes const& h)
    : task{"way_matches",
           data_path,
           c,
           {h.tt_,          h.osm_,          osr_version(),
            n_version(),    matches_version(), way_matches_version(),
            get_way_matches_hash(c)}} {}

way_matches_import::~way_matches_import() = default;

void way_matches_import::run() {
  auto d = data{data_path_, false};
  d.load_osr();
  d.load_tt(c_.osr_footpath_ ? "tt_ext.bin" : "tt.bin");
  d.load_matches();

  utl::get_active_progress_tracker()
      ->status("Prepare Platform Way Matches")
      .out_bounds(0, 100);

  auto way_matches = std::make_unique<way_matches_storage>(
      data_path_, cista::mmap::protection::WRITE,
      c_.timetable_->preprocess_max_matching_distance_);
  way_matches->preprocess_osr_matches(*d.tt_, *d.pl_, *d.w_, *d.l_,
                                      *d.matches_);
}

bool way_matches_import::is_enabled() const {
  return c_.timetable_ && c_.use_street_routing() &&
         c_.timetable_->preprocess_max_matching_distance_ > 0.0;
}

}  // namespace motis
