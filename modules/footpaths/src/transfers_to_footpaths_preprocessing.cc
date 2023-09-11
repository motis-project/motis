#include "motis/footpaths/transfers_to_footpaths_preprocessing.h"

#include "utl/enumerate.h"
#include "utl/progress_tracker.h"

#include "motis/footpaths/keys.h"

namespace n = ::nigiri;

namespace motis::footpaths {

hash_map<nlocation_key_t, n::location_idx_t> to_location_key_to_idx_(
    n::vector_map<n::location_idx_t, geo::latlng> coords) {
  auto res = hash_map<nlocation_key_t, n::location_idx_t>{};

  for (auto const [i, coord] : utl::enumerate(coords)) {
    res.insert(std::pair<nlocation_key_t, n::location_idx_t>(
        to_key(coord), n::location_idx_t{i}));
  }
  return res;
}

preprocessed_transfers to_preprocessed_footpaths(
    transfer_preprocessing_data const& data) {
  auto fp = preprocessed_transfers{};

  auto const location_key_to_idx = to_location_key_to_idx_(data.coords_);

  // update progress tracker
  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->reset_bounds().in_high(data.transfer_results_.size());

  // initialize out/in multimap
  for (auto prf_idx = n::profile_idx_t{0U}; prf_idx < n::kMaxProfiles;
       ++prf_idx) {
    for (auto loc_idx = n::location_idx_t{0U}; loc_idx < data.coords_.size();
         ++loc_idx) {
      fp.out_[prf_idx].emplace_back();
    }
  }

  for (auto prf_idx = n::profile_idx_t{0U}; prf_idx < n::kMaxProfiles;
       ++prf_idx) {
    for (auto loc_idx = n::location_idx_t{0U}; loc_idx < data.coords_.size();
         ++loc_idx) {
      fp.in_[prf_idx].emplace_back();
    }
  }

  auto const& single_update = [&](transfer_result const& tres) {
    progress_tracker->increment();

    for (auto [to_nloc, info] : utl::zip(tres.to_nloc_keys_, tres.infos_)) {
      if (data.profiles_.count(
              data.profile_key_to_profile_name.at(tres.profile_)) == 0 ||
          location_key_to_idx.count(tres.from_nloc_key_) == 0 ||
          location_key_to_idx.count(to_nloc) == 0) {
        continue;
      }

      auto const prf_idx =
          data.profiles_.at(data.profile_key_to_profile_name.at(tres.profile_));
      auto const from_idx = location_key_to_idx.at(tres.from_nloc_key_);
      auto const to_idx = location_key_to_idx.at(to_nloc);

      fp.out_[prf_idx][from_idx].emplace_back(
          n::footpath{to_idx, info.duration_});
      fp.in_[prf_idx][to_idx].emplace_back(
          n::footpath{from_idx, info.duration_});
    }
  };

  for (auto const& tr : data.transfer_results_) {
    single_update(tr);
  }

  return fp;
}

}  // namespace motis::footpaths
