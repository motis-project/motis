#include "motis/transfers/storage/to_nigiri.h"

#include "utl/enumerate.h"
#include "utl/progress_tracker.h"
#include "utl/zip.h"

namespace n = ::nigiri;

namespace motis::transfers {

// Returns an mapping from nigiri location key to corresponding nigiri location
// idx in the used timetable.
hash_map<nlocation_key_t, n::location_idx_t> to_location_key_to_idx_(
    n::vector_map<n::location_idx_t, geo::latlng> coords) {
  auto res = hash_map<nlocation_key_t, n::location_idx_t>{};

  for (auto const [i, coord] : utl::enumerate(coords)) {
    res.insert(std::pair<nlocation_key_t, n::location_idx_t>(
        to_key(coord), n::location_idx_t{i}));
  }
  return res;
}

nigiri_transfers build_nigiri_transfers(to_nigiri_data const& data) {
  auto ntransfers = nigiri_transfers{};

  auto const location_key_to_idx = to_location_key_to_idx_(data.coords_);

  // update progress tracker
  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->reset_bounds().in_high(data.transfer_results_.size());

  // initialize out_/in_ multimap in ntransfers
  for (auto prf_idx = n::profile_idx_t{0U}; prf_idx < n::kMaxProfiles;
       ++prf_idx) {
    for (auto loc_idx = n::location_idx_t{0U}; loc_idx < data.coords_.size();
         ++loc_idx) {
      ntransfers.out_[prf_idx].emplace_back();
    }
  }

  for (auto prf_idx = n::profile_idx_t{0U}; prf_idx < n::kMaxProfiles;
       ++prf_idx) {
    for (auto loc_idx = n::location_idx_t{0U}; loc_idx < data.coords_.size();
         ++loc_idx) {
      ntransfers.in_[prf_idx].emplace_back();
    }
  }

  // define single `transfer_result` to ntransfers update
  auto const& single_update = [&](transfer_result const& tres) {
    progress_tracker->increment();

    for (auto [to_nloc, info] : utl::zip(tres.to_nloc_keys_, tres.infos_)) {
      if (data.profile_name_to_tt_idx_.count(
              data.profile_key_to_profile_name_.at(tres.profile_)) == 0 ||
          location_key_to_idx.count(tres.from_nloc_key_) == 0 ||
          location_key_to_idx.count(to_nloc) == 0) {
        continue;
      }

      auto const prf_idx = data.profile_name_to_tt_idx_.at(
          data.profile_key_to_profile_name_.at(tres.profile_));
      auto const from_idx = location_key_to_idx.at(tres.from_nloc_key_);
      auto const to_idx = location_key_to_idx.at(to_nloc);

      ntransfers.out_[prf_idx][from_idx].emplace_back(
          n::footpath{to_idx, info.duration_});
      ntransfers.in_[prf_idx][to_idx].emplace_back(
          n::footpath{from_idx, info.duration_});
    }
  };

  for (auto const& tr : data.transfer_results_) {
    single_update(tr);
  }

  return ntransfers;
}

}  // namespace motis::transfers
