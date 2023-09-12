#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>

#include "motis/footpaths/matching.h"
#include "motis/footpaths/platform/platform_index.h"
#include "motis/footpaths/storage/database.h"
#include "motis/footpaths/transfer/transfer_request.h"
#include "motis/footpaths/transfer/transfer_result.h"
#include "motis/footpaths/transfers_to_footpaths_preprocessing.h"
#include "motis/footpaths/types.h"

#include "motis/ppr/profile_info.h"

namespace motis::footpaths {

enum class data_request_type { kPartialOld, kPartialUpdate, kFull };

struct storage {

  storage(std::filesystem::path const& db_file_path,
          std::size_t const db_max_size)
      : db_{db_file_path, db_max_size} {}

  // Initializes the storage for the footpath module.
  void initialize(set<profile_key_t> const&,
                  hash_map<profile_key_t, ppr::profile_info> const&);

  // Returns a `matching_data` struct containing all the data used during
  // matching nigiri locations and osm extracted platforms.
  matching_data get_matching_data(::nigiri::timetable const&);

  // Returns a map of all known matchings of nigiri locations to osm extracted
  // platforms. Combines old and new matchings.
  hash_map<nlocation_key_t, platform> get_all_matchings();

  // Returns whether the storage contains `transfer_requests_keys` for the
  // corresponding `data_request_type`.
  bool has_transfer_requests_keys(data_request_type const);

  // Returns for the given `data_request_type` a list of
  // `transfer_requests_keys` which are stored in the storage.
  transfer_requests_keys get_transfer_requests_keys(data_request_type const);

  // Returns a `treq_k_generation_data` struct containing all the data used
  // during the generation of `transfer_request_keys`.
  treq_k_generation_data get_transfer_request_keys_generation_data();

  // Returns a `transfer_preprocessing_data` struct containing all the data used
  // during the transfer preprocessing of `transfer_results`.
  transfer_preprocessing_data get_transfer_preprocessing_data(
      ::nigiri::timetable const&);

  // --- public db api ---

  // Adds new profile names to the database. Assigns a unique
  // `profile_key_t` to previously unknown profile names. Updates the `name
  // -> profile_key_t` storage map. Updates the `profile_key_t -> name`
  // storage map.
  void add_new_profiles(std::vector<string> const&);

  // Adds new platforms to the database. Previously unknown platforms are added
  // to the `update_state_` state struct. Deletes old `update_state_` platforms.
  void add_new_platforms(platforms&);

  // Adds new matching results to the database. Previously unknown matches are
  // added to the `update_state_` state struct.
  void add_new_matching_results(matching_results const&);

  // Adds new transfer requests keys to the database. Previously unknown
  // transfer requests keys are added to the `update_state_` state struct.
  // Previously known transfer requests keys are updated. The difference
  // (described by the given transfer_request_keys is added to the
  // `update_state_` state struct.
  // Update merges the old transfer request keys with the new one.
  void add_new_transfer_requests_keys(transfer_requests_keys const&);

  // Adds new transfer results to the database. Previously unknown transfer
  // results are added to the `update_state_` state struct. Previously known
  // transfer results are updated. The difference (described in the transfer
  // result struct is added to the `update_state_` struct.
  // Update merges the old transfer result with the new one.
  void add_new_transfer_results(transfer_results const&);

  hash_map<string, profile_key_t> profile_name_to_profile_key_;
  hash_map<profile_key_t, string> profile_key_to_profile_name_;

private:
  // Loads all footpaths data from the database and stores it in the
  // `old_state_` state struct.
  void load_old_state_from_db(set<profile_key_t> const&);

  struct state {
    std::unique_ptr<platform_index> pfs_idx_;
    std::unique_ptr<platform_index> matched_pfs_idx_;

    bool set_matched_pfs_idx_{false};
    bool set_pfs_idx_{false};

    // matched nigiri location keys
    vector<nlocation_key_t> nloc_keys_;

    // mapping matched nloc to pf
    hash_map<nlocation_key_t, platform> matches_;
    transfer_requests_keys transfer_requests_keys_;
    transfer_results transfer_results_;
  } old_state_, update_state_;

  set<profile_key_t> used_profiles_;
  hash_map<profile_key_t, ppr::profile_info> profile_key_to_profile_info_;
  database db_;
};

}  // namespace motis::footpaths
