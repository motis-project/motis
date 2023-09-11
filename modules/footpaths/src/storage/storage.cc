#include "motis/footpaths/storage/storage.h"

#include <utility>

#include "motis/footpaths/keys.h"

#include "utl/pipes/all.h"
#include "utl/pipes/transform.h"
#include "utl/pipes/vec.h"

namespace motis::footpaths {

void storage::initialize(set<profile_key_t> const& used_profiles,
                         hash_map<profile_key_t, ppr::profile_info> const&
                             profile_key_to_profile_info) {
  used_profiles_ = used_profiles;
  load_old_state_from_db(used_profiles_);
  profile_key_to_profile_info_.insert(profile_key_to_profile_info.begin(),
                                      profile_key_to_profile_info.end());
}

matching_data storage::get_matching_data(::nigiri::timetable const& tt) {
  return {tt.locations_, old_state_.matches_, *(old_state_.pfs_idx_),
          *(update_state_.pfs_idx_), update_state_.set_pfs_idx_};
}

hash_map<nlocation_key_t, platform> storage::get_all_matchings() {
  auto all_matchings = old_state_.matches_;
  all_matchings.insert(update_state_.matches_.begin(),
                       update_state_.matches_.end());
  return all_matchings;
}

bool storage::has_transfer_requests_keys(data_request_type req_type) {
  return get_transfer_requests_keys(req_type).empty();
}

transfer_requests_keys storage::get_transfer_requests_keys(
    data_request_type const req_type) {
  switch (req_type) {
    case data_request_type::kPartialOld:
      return old_state_.transfer_requests_keys_;
    case data_request_type::kPartialUpdate:
      return update_state_.transfer_requests_keys_;
    case data_request_type::kFull:
      auto full = old_state_.transfer_requests_keys_;
      full.insert(full.end(), update_state_.transfer_requests_keys_.begin(),
                  update_state_.transfer_requests_keys_.end());
      return full;
  }
}

treq_k_generation_data storage::get_transfer_request_keys_generation_data() {
  return {{*(old_state_.matched_pfs_idx_), old_state_.nloc_keys_,
           old_state_.set_matched_pfs_idx_},
          {*(update_state_.matched_pfs_idx_), update_state_.nloc_keys_,
           update_state_.set_matched_pfs_idx_},
          profile_key_to_profile_info_};
}

transfer_preprocessing_data storage::get_transfer_preprocessing_data(
    ::nigiri::timetable const& tt) {
  auto tress = db_.get_transfer_results(used_profiles_);
  return {tt.locations_.coordinates_, tt.profiles_,
          profile_key_to_profile_name_, tress};
}

void storage::add_new_profiles(std::vector<string> const& profile_names) {
  db_.put_profiles(profile_names);
  profile_name_to_profile_key_ = db_.get_profile_keys();
  profile_key_to_profile_name_ = db_.get_profile_key_to_name();
}

void storage::add_new_platforms(platforms& pfs) {
  auto const added_to_db = db_.put_platforms(pfs);
  auto new_pfs = utl::all(added_to_db) |
                 utl::transform([&pfs](auto const i) { return pfs[i]; }) |
                 utl::vec();

  update_state_.pfs_idx_ =
      std::make_unique<platform_index>(platform_index{new_pfs});
  update_state_.set_pfs_idx_ = true;
}

void storage::add_new_matching_results(matching_results const& mrs) {
  auto const added_to_db = db_.put_matching_results(mrs);
  auto const new_mrs = utl::all(added_to_db) |
                       utl::transform([&mrs](auto const i) { return mrs[i]; }) |
                       utl::vec();

  auto matched_pfs = platforms{};
  for (auto const& mr : new_mrs) {
    update_state_.matches_.insert(
        std::pair<nlocation_key_t, platform>(to_key(mr.nloc_pos_), mr.pf_));
    update_state_.nloc_keys_.emplace_back(to_key(mr.nloc_pos_));
    matched_pfs.emplace_back(mr.pf_);
  }

  update_state_.matched_pfs_idx_ =
      std::make_unique<platform_index>(platform_index{matched_pfs});
  update_state_.set_matched_pfs_idx_ = true;
}

void storage::add_new_transfer_requests_keys(
    transfer_requests_keys const& treqs_k) {
  auto const updated_in_db = db_.update_transfer_requests_keys(treqs_k);
  auto const added_to_db = db_.put_transfer_requests_keys(treqs_k);

  auto const updated_treqs_k =
      utl::all(updated_in_db) |
      utl::transform([&treqs_k](auto const i) { return treqs_k[i]; }) |
      utl::vec();
  auto const new_treqs_k =
      utl::all(added_to_db) |
      utl::transform([&treqs_k](auto const i) { return treqs_k[i]; }) |
      utl::vec();

  auto result = transfer_requests_keys{};
  result.insert(result.end(), updated_treqs_k.begin(), updated_treqs_k.end());
  result.insert(result.end(), new_treqs_k.begin(), new_treqs_k.end());

  update_state_.transfer_requests_keys_ = result;
}

void storage::add_new_transfer_results(transfer_results const& tres) {
  auto const updated_in_db = db_.update_transfer_results(tres);
  auto const added_to_db = db_.put_transfer_results(tres);

  auto const updated_tres =
      utl::all(updated_in_db) |
      utl::transform([&tres](auto const i) { return tres[i]; }) | utl::vec();
  auto const new_tres =
      utl::all(added_to_db) |
      utl::transform([&tres](auto const i) { return tres[i]; }) | utl::vec();

  auto result = transfer_results{};
  result.insert(result.end(), updated_tres.begin(), updated_tres.end());
  result.insert(result.end(), new_tres.begin(), new_tres.end());

  update_state_.transfer_results_ = result;
}

void storage::load_old_state_from_db(set<profile_key_t> const& profile_keys) {
  auto old_pfs = db_.get_platforms();
  old_state_.pfs_idx_ =
      std::make_unique<platform_index>(platform_index{old_pfs});
  old_state_.set_pfs_idx_ = true;
  old_state_.matches_ = db_.get_loc_to_pf_matchings();
  old_state_.transfer_requests_keys_ =
      db_.get_transfer_requests_keys(profile_keys);
  old_state_.transfer_results_ = db_.get_transfer_results(profile_keys);

  auto matched_pfs = platforms{};
  auto matched_nloc_keys = vector<nlocation_key_t>{};
  for (auto const& [k, pf] : old_state_.matches_) {
    matched_nloc_keys.emplace_back(k);
    matched_pfs.emplace_back(pf);
  }
  old_state_.nloc_keys_ = matched_nloc_keys;
  old_state_.matched_pfs_idx_ =
      std::make_unique<platform_index>(platform_index{matched_pfs});
  old_state_.set_matched_pfs_idx_ = true;
}

}  // namespace motis::footpaths
