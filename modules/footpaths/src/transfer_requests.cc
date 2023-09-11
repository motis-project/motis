#include "motis/footpaths/transfer_requests.h"

#include "motis/footpaths/keys.h"
#include "motis/footpaths/types.h"

namespace motis::footpaths {

transfer_requests to_transfer_requests(
    transfer_requests_keys const& treqs_k,
    hash_map<nlocation_key_t, platform> const& matches) {
  auto treqs = transfer_requests{};

  for (auto const& treq_k : treqs_k) {
    auto treq = transfer_request{};

    treq.from_nloc_key_ = treq_k.from_nloc_key_;
    treq.profile_ = treq_k.profile_;

    // extract from_pf
    treq.transfer_start_ = matches.at(treq_k.from_nloc_key_);

    // extract to_pfs
    for (auto to_nloc_key : treq_k.to_nloc_keys_) {
      treq.transfer_targets_.emplace_back(matches.at(to_nloc_key));
      treq.to_nloc_keys_.emplace_back(to_nloc_key);
    }

    treqs.emplace_back(treq);
  }

  return treqs;
}

/**
 * old_to_old: build transfer requests from already processed (matched
 * platforms) in old_state; use if profiles_hash has been changed
 */
transfer_requests_keys generate_transfer_requests_keys(
    state const& old_state, state const& update_state,
    hash_map<profile_key_t, ppr::profile_info> const& profiles,
    bool const old_to_old) {
  auto result = transfer_requests_keys{};

  auto const all_pairs_trs = [&](state const& from_state, state const& to_state,
                                 profile_key_t const& prf_key) {
    auto from_to_trs = transfer_requests_keys{};
    auto const& pi = profiles.at(prf_key);
    auto prf_dist = pi.profile_.walking_speed_ * pi.profile_.duration_limit_;

    for (auto i = std::size_t{0}; i < from_state.matched_pfs_idx_->size();
         ++i) {

      auto from_pf = from_state.matched_pfs_idx_->get_platform(i);
      auto target_ids =
          to_state.matched_pfs_idx_->get_other_platforms_in_radius(from_pf,
                                                                   prf_dist);
      auto from_nloc_key = from_state.nloc_keys_[i];

      if (target_ids.empty()) {
        continue;
      }

      auto tmp = transfer_request_keys{};
      auto to_nloc_keys = vector<nlocation_key_t>{};

      for (auto t_id : target_ids) {
        to_nloc_keys.emplace_back(to_state.nloc_keys_[t_id]);
      }

      tmp.from_nloc_key_ = from_nloc_key;
      tmp.to_nloc_keys_ = to_nloc_keys;
      tmp.profile_ = prf_key;

      from_to_trs.emplace_back(tmp);
    }

    return from_to_trs;
  };

  // new possible transfers: 1 -> 2, 2 -> 1, 2 -> 2
  for (auto const& [prf_key, prf_info] : profiles) {
    if (old_to_old) {
      auto trs11 = all_pairs_trs(old_state, old_state, prf_key);
      result.insert(result.end(), trs11.begin(), trs11.end());
    }

    if (!update_state.set_matched_pfs_idx_) {
      continue;
    }

    // new transfers from old to update (1 -> 2)
    auto trs12 = all_pairs_trs(old_state, update_state, prf_key);
    // new transfers from update to old (2 -> 1)
    auto trs21 = all_pairs_trs(update_state, old_state, prf_key);
    // new transfers from update to update (2 -> 2)
    auto trs22 = all_pairs_trs(update_state, update_state, prf_key);

    result.insert(result.end(), trs12.begin(), trs12.end());
    result.insert(result.end(), trs21.begin(), trs21.end());
    result.insert(result.end(), trs22.begin(), trs22.end());
  }

  return result;
}

}  // namespace motis::footpaths
