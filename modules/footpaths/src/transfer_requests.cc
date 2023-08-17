#include "motis/footpaths/transfer_requests.h"

#include <cstddef>
#include <vector>

#include "motis/footpaths/platforms.h"

#include "motis/footpaths/types.h"

namespace motis::footpaths {

transfer_requests to_transfer_requests(transfer_requests_keys const& treqs_k,
                                       database& db) {
  auto treqs = transfer_requests{};
  auto pfs_with_key = db.get_platforms_with_key();

  auto has_pf_key = [&](string const& pf_key) {
    return pfs_with_key.count(pf_key) == 1;
  };

  for (auto const& treq_k : treqs_k) {
    auto not_found = 0U;
    auto treq = transfer_request{};

    treq.from_nloc_key_ = treq_k.from_nloc_key_;
    treq.profile_ = treq_k.profile_;

    assert(treq_k.to_nloc_keys_.size() == treq_k.to_pf_keys_.size());

    // extract from_pf
    if (has_pf_key(treq_k.from_pf_key_)) {
      treq.transfer_start_ = pfs_with_key.at(treq_k.from_pf_key_);
    } else {
      ++not_found;
    }

    // extract to_pfs
    for (auto i = 0; i < treq_k.to_pf_keys_.size(); ++i) {
      auto to_pf_key = treq_k.to_pf_keys_[i];
      auto to_nloc_key = treq_k.to_nloc_keys_[i];

      if (has_pf_key(to_pf_key)) {
        treq.transfer_targets_.emplace_back(pfs_with_key.at(to_pf_key));
        treq.to_nloc_keys_.emplace_back(to_nloc_key);
      } else {
        ++not_found;
      }
    }

    if (not_found != 0) {
      continue;
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
    std::map<std::string, ppr::profile_info> const& profiles,
    bool const old_to_old) {
  auto result = transfer_requests_keys{};

  auto const all_pairs_trs = [&](state const& from_state, state const& to_state,
                                 std::string const& profile) {
    auto from_to_trs = transfer_requests_keys{};
    auto const& pi = profiles.at(profile);
    auto prf_dist = pi.profile_.walking_speed_ * pi.profile_.duration_limit_;

    for (auto i = std::size_t{0}; i < from_state.matched_pfs_idx_->size();
         ++i) {

      auto from_pf = from_state.matched_pfs_idx_->get_platform(i);
      auto from_pf_key = to_key(from_pf);
      auto target_ids =
          to_state.matched_pfs_idx_->valid_in_radius(from_pf, prf_dist);
      auto from_nloc_key = from_state.nloc_keys[i];

      if (target_ids.empty()) {
        continue;
      }

      auto tmp = transfer_request_keys{};
      auto to_pf_keys = strings{};
      auto to_nloc_keys = strings{};

      for (auto t_id : target_ids) {
        to_pf_keys.emplace_back(
            to_key(to_state.matched_pfs_idx_->get_platform(t_id)));
      }

      for (auto t_id : target_ids) {
        to_nloc_keys.emplace_back(to_state.nloc_keys[t_id]);
      }

      tmp.from_pf_key_ = from_pf_key;
      tmp.from_nloc_key_ = from_nloc_key;
      tmp.to_pf_keys_ = to_pf_keys;
      tmp.to_nloc_keys_ = to_nloc_keys;
      tmp.profile_ = string{profile};

      from_to_trs.emplace_back(tmp);
    }

    return from_to_trs;
  };

  // new possible transfers: 1 -> 2, 2 -> 1, 2 -> 2
  for (auto const& [prf_name, prf_info] : profiles) {
    if (old_to_old) {
      auto trs11 = all_pairs_trs(old_state, old_state, prf_name);
      result.insert(result.end(), trs11.begin(), trs11.end());
    }

    if (!update_state.set_matched_pfs_idx_) {
      continue;
    }

    // new transfers from old to update (1 -> 2)
    auto trs12 = all_pairs_trs(old_state, update_state, prf_name);
    // new transfers from update to old (2 -> 1)
    auto trs21 = all_pairs_trs(update_state, old_state, prf_name);
    // new transfers from update to update (2 -> 2)
    auto trs22 = all_pairs_trs(update_state, update_state, prf_name);

    result.insert(result.end(), trs12.begin(), trs12.end());
    result.insert(result.end(), trs21.begin(), trs21.end());
    result.insert(result.end(), trs22.begin(), trs22.end());
  }

  return result;
}

}  // namespace motis::footpaths
