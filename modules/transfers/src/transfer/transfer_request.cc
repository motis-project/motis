#include "motis/transfers/transfer/transfer_request.h"

#include "motis/transfers/types.h"

#include "fmt/core.h"

#include "utl/verify.h"

namespace motis::transfers {

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

transfer_requests_keys generate_transfer_requests_keys(
    treq_k_generation_data const& data, transfer_request_options const& opts) {
  auto result = transfer_requests_keys{};
  auto const profiles = data.profile_key_to_profile_info_;

  auto const all_pairs_trs =
      [&profiles](treq_k_generation_data::matched_nloc_pf_data const& from,
                  treq_k_generation_data::matched_nloc_pf_data const& to,
                  profile_key_t const& prf_key) {
        auto from_to_trs = transfer_requests_keys{};
        auto const& pi = profiles.at(prf_key);
        auto prf_dist =
            pi.profile_.walking_speed_ * pi.profile_.duration_limit_;

        if (from.matched_pfs_idx_.size() == 0 ||
            to.matched_pfs_idx_.size() == 0) {
          return from_to_trs;
        }

        for (auto i = std::size_t{0}; i < from.matched_pfs_idx_.size(); ++i) {

          auto from_pf = from.matched_pfs_idx_.get_platform(i);
          auto target_ids = to.matched_pfs_idx_.get_other_platforms_in_radius(
              from_pf, prf_dist);
          auto from_nloc_key = from.nloc_keys_[i];

          if (target_ids.empty()) {
            continue;
          }

          auto tmp = transfer_request_keys{};
          auto to_nloc_keys = vector<nlocation_key_t>{};

          for (auto t_id : target_ids) {
            to_nloc_keys.emplace_back(to.nloc_keys_[t_id]);
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
    if (opts.old_to_old_) {
      auto trs11 = all_pairs_trs(data.old_, data.old_, prf_key);
      result.insert(result.end(), trs11.begin(), trs11.end());
    }

    if (!data.update_.set_matched_pfs_idx_) {
      continue;
    }

    // new transfers from old to update (1 -> 2)
    auto trs12 = all_pairs_trs(data.old_, data.update_, prf_key);
    // new transfers from update to old (2 -> 1)
    auto trs21 = all_pairs_trs(data.update_, data.old_, prf_key);
    // new transfers from update to update (2 -> 2)
    auto trs22 = all_pairs_trs(data.update_, data.update_, prf_key);

    result.insert(result.end(), trs12.begin(), trs12.end());
    result.insert(result.end(), trs21.begin(), trs21.end());
    result.insert(result.end(), trs22.begin(), trs22.end());
  }

  return result;
}

transfer_request_keys merge(transfer_request_keys const& lhs,
                            transfer_request_keys const& rhs) {
  auto merged = transfer_request_keys{};
  auto added_to_nlocs = set<nlocation_key_t>{};

  utl::verify(
      lhs.from_nloc_key_ == rhs.from_nloc_key_,
      "Cannot merge two transfer requests from different nigiri locations.");
  utl::verify(lhs.profile_ == rhs.profile_,
              "Cannot merge two transfer requests with different profiles");

  merged.from_nloc_key_ = lhs.from_nloc_key_;
  merged.profile_ = lhs.profile_;

  merged.to_nloc_keys_ = lhs.to_nloc_keys_;

  // build added_keys set
  for (auto const& nloc_key : merged.to_nloc_keys_) {
    added_to_nlocs.insert(nloc_key);
  }

  // insert new and unique nloc/pf keys
  for (auto nloc_key : rhs.to_nloc_keys_) {
    if (added_to_nlocs.count(nloc_key) == 1) {
      continue;
    }

    merged.to_nloc_keys_.emplace_back(nloc_key);
    added_to_nlocs.insert(nloc_key);
  }

  return merged;
}

string to_key(transfer_request_keys const& treq_k) {
  return {fmt::format("{}{}", treq_k.from_nloc_key_, treq_k.profile_)};
}

string to_key(transfer_request const& treq) {
  return {fmt::format("{}{}", treq.from_nloc_key_, treq.profile_)};
}

std::ostream& operator<<(std::ostream& out, transfer_request const& treq) {
  auto treq_repr = fmt::format("[transfer request] {} has {} locations.",
                               to_key(treq), treq.to_nloc_keys_.size());
  return out << treq_repr;
}

std::ostream& operator<<(std::ostream& out,
                         transfer_request_keys const& treq_k) {
  auto treq_k_repr = fmt::format("[transfer request keys] {} has {} locations.",
                                 to_key(treq_k), treq_k.to_nloc_keys_.size());
  return out << treq_k_repr;
}

}  // namespace motis::transfers
