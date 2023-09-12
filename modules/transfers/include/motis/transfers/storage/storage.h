#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>

#include "motis/transfers/matching.h"
#include "motis/transfers/platform/platform_index.h"
#include "motis/transfers/storage/database.h"
#include "motis/transfers/storage/to_nigiri.h"
#include "motis/transfers/transfer/transfer_request.h"
#include "motis/transfers/transfer/transfer_result.h"
#include "motis/transfers/types.h"

#include "motis/ppr/profile_info.h"

#include "nigiri/timetable.h"

namespace motis::transfers {

enum class data_request_type { kPartialOld, kPartialUpdate, kFull };

struct storage {

  storage(std::filesystem::path const& db_file_path,
          std::size_t const db_max_size, ::nigiri::timetable& tt)
      : tt_(tt), db_{db_file_path, db_max_size} {}

  // Initializes the storage for the footpath module.
  void initialize(set<profile_key_t> const&,
                  hash_map<profile_key_t, ppr::profile_info> const&);

  // Saves the current timetable in the specified file path.
  void save_tt(std::filesystem::path const&);

  // Updates the `nigiri::timetable` with the transfer results stored in the
  // storage. (considers `old_state_` data and `update_state_` data).
  void update_tt(std::filesystem::path const&);

  // Returns a `matching_data` struct containing all the data used during
  // matching nigiri locations and osm extracted platforms.
  matching_data get_matching_data();

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

  ::nigiri::timetable& tt_;

  hash_map<string, profile_key_t> profile_name_to_profile_key_;
  hash_map<profile_key_t, string> profile_key_to_profile_name_;

private:
  // Loads all transfers data from the database and stores it in the
  // `old_state_` state struct.
  void load_old_state_from_db(set<profile_key_t> const&);

  // Returns a `to_nigiri_data` struct containing all the data used
  // during the transfer preprocessing of `transfer_results`.
  to_nigiri_data get_to_nigiri_data();

  // Deletes all transfers from the timetable.
  void reset_timetable_transfers();

  // Adds the given transfers to the `nigiri::timetable` and removes currently
  // existing transfers (in a `nigiri_transfers` struct) from the
  // `nigiri::timetable` for this purpose.
  void set_new_timetable_transfers(nigiri_transfers const&);

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

}  // namespace motis::transfers
