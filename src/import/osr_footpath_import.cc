#include "motis/import/osr_footpath_import.h"

#include "cista/hashing.h"
#include "cista/io.h"

#include "utl/helpers/algorithm.h"

#include "nigiri/clasz.h"

#include "motis/compute_footpaths.h"
#include "motis/data.h"

using std::chrono_literals::operator""h;
using std::chrono_literals::operator""min;

namespace n = nigiri;

namespace motis {

namespace fs = std::filesystem;

meta_entry_t get_osr_footpath_settings_hash(config const& c) {
  auto osr_footpath_settings_hash =
      meta_entry_t{"osr_footpath_settings", cista::BASE_HASH};
  if (c.timetable_) {
    auto& h = osr_footpath_settings_hash.second;
    h = cista::hash_combine(h, c.timetable_->use_osm_stop_coordinates_,
                            c.timetable_->extend_missing_footpaths_,
                            c.timetable_->max_matching_distance_,
                            c.timetable_->max_footpath_length_);
  }
  return osr_footpath_settings_hash;
}

osr_footpath_import::osr_footpath_import(fs::path const& data_path,
                                         config const& c,
                                         dataset_hashes const& h)
    : task{"osr_footpath",
           data_path,
           c,
           {h.tt_, h.osm_, get_osr_footpath_settings_hash(c), osr_version(),
            osr_footpath_version(), n_version()}} {}

osr_footpath_import::~osr_footpath_import() = default;

void osr_footpath_import::run() {
  auto d = data{data_path_, false};
  d.load_osr();
  d.load_tt("tt.bin");

  auto const profiles = std::vector<routed_transfers_settings>{
      {.profile_ = osr::search_profile::kFoot,
       .profile_idx_ = n::kFootProfile,
       .max_matching_distance_ = c_.timetable_->max_matching_distance_,
       .extend_missing_ = c_.timetable_->extend_missing_footpaths_,
       .max_duration_ = c_.timetable_->max_footpath_length_ * 1min},
      {.profile_ = osr::search_profile::kWheelchair,
       .profile_idx_ = n::kWheelchairProfile,
       .max_matching_distance_ = 8.0,
       .max_duration_ = c_.timetable_->max_footpath_length_ * 1min},
      {.profile_ = osr::search_profile::kCar,
       .profile_idx_ = n::kCarProfile,
       .max_matching_distance_ = 250.0,
       .max_duration_ = 8h,
       .is_candidate_ = [&](n::location_idx_t const l) {
         return utl::any_of(d.tt_->location_routes_[l], [&](auto r) {
           return d.tt_->has_car_transport(r);
         });
       }}};
  auto const elevator_footpath_map =
      compute_footpaths(*d.w_, *d.l_, *d.pl_, *d.tt_, d.elevations_.get(),
                        c_.timetable_->use_osm_stop_coordinates_, profiles);

  cista::write(data_path_ / "elevator_footpath_map.bin", elevator_footpath_map);
  d.tt_->write(data_path_ / "tt_ext.bin");
}

bool osr_footpath_import::is_enabled() const { return c_.osr_footpath_; }

}  // namespace motis
