#include "motis/footpaths/transfers.h"

#include <sstream>

#include "fmt/core.h"

#include "motis/footpaths/types.h"

#include "utl/verify.h"
#include "utl/zip.h"

namespace motis::footpaths {

transfer_request_keys merge(transfer_request_keys const& treq_k_a,
                            transfer_request_keys const& treq_k_b) {
  auto merged = transfer_request_keys{};
  auto added_keys = set<string>{};

  utl::verify(
      treq_k_a.from_nloc_key_ == treq_k_b.from_nloc_key_,
      "Cannot merge two transfer requests from different nigiri locations.");
  utl::verify(treq_k_a.from_pf_key_ == treq_k_b.from_pf_key_,
              "Cannot merge two transfer requests from different platforms.");
  utl::verify(treq_k_a.profile_ == treq_k_b.profile_,
              "Cannot merge two transfer requests with different profiles");
  utl::verify(
      treq_k_a.to_nloc_keys_.size() == treq_k_a.to_pf_keys_.size(),
      "(A) Cannot merge transfer requests with invalid nigiri and platform "
      "matching.");
  utl::verify(
      treq_k_b.to_nloc_keys_.size() == treq_k_b.to_pf_keys_.size(),
      "(B) Cannot merge transfer requests with invalid nigiri and platform "
      "matching.");

  merged.from_nloc_key_ = treq_k_a.from_nloc_key_;
  merged.from_pf_key_ = treq_k_a.from_pf_key_;
  merged.profile_ = treq_k_a.profile_;

  merged.to_nloc_keys_ = treq_k_a.to_nloc_keys_;
  merged.to_pf_keys_ = treq_k_a.to_pf_keys_;

  // build added_keys hash_map
  for (const auto& nloc_key : merged.to_nloc_keys_) {
    added_keys.insert(nloc_key);
  }

  // insert new and unique nloc/pf keys
  for (auto [nloc_key, pf_key] :
       utl::zip(treq_k_b.to_nloc_keys_, treq_k_b.to_pf_keys_)) {
    if (added_keys.count(nloc_key) == 0) {
      merged.to_nloc_keys_.emplace_back(nloc_key);
      merged.to_pf_keys_.emplace_back(pf_key);

      added_keys.insert(nloc_key);
    }
  }

  return merged;
}

std::ostream& operator<<(std::ostream& out, transfer_info const& tinfo) {
  auto tinfo_repr =
      fmt::format("dur: {}, dist: {}", tinfo.duration_, tinfo.distance_);
  return out << tinfo_repr;
}

std::ostream& operator<<(std::ostream& out, transfer_result const& tres) {
  std::stringstream tres_repr;
  tres_repr << "[transfer result] " << to_key(tres) << ": " << tres.info_;
  return out << tres_repr.str();
}

std::ostream& operator<<(std::ostream& out, transfer_request const& treq) {
  auto treq_repr = fmt::format(
      "[transfer request] {} has {} locations and {} platforms.", to_key(treq),
      treq.to_nloc_keys_.size(), treq.to_nloc_keys_.size());
  return out << treq_repr;
}

std::ostream& operator<<(std::ostream& out,
                         transfer_request_keys const& treq_k) {
  auto treq_k_repr = fmt::format(
      "[transfer request keys] {} has {} locations and {} platforms.",
      to_key(treq_k), treq_k.to_nloc_keys_.size(), treq_k.to_pf_keys_.size());
  return out << treq_k_repr;
}

}  // namespace motis::footpaths