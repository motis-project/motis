#include "motis/footpaths/transfers.h"

#include <sstream>

#include "fmt/core.h"

#include "motis/footpaths/keys.h"
#include "motis/footpaths/types.h"

#include "utl/verify.h"
#include "utl/zip.h"

namespace motis::footpaths {

transfer_request_keys merge(transfer_request_keys const& treq_k_a,
                            transfer_request_keys const& treq_k_b) {
  auto merged = transfer_request_keys{};
  auto added_to_nlocs = set<key64_t>{};

  utl::verify(
      treq_k_a.from_nloc_key_ == treq_k_b.from_nloc_key_,
      "Cannot merge two transfer requests from different nigiri locations.");
  utl::verify(treq_k_a.profile_ == treq_k_b.profile_,
              "Cannot merge two transfer requests with different profiles");

  merged.from_nloc_key_ = treq_k_a.from_nloc_key_;
  merged.profile_ = treq_k_a.profile_;

  merged.to_nloc_keys_ = treq_k_a.to_nloc_keys_;

  // build added_keys set
  for (auto const& nloc_key : merged.to_nloc_keys_) {
    added_to_nlocs.insert(nloc_key);
  }

  // insert new and unique nloc/pf keys
  for (auto nloc_key : treq_k_b.to_nloc_keys_) {
    if (added_to_nlocs.count(nloc_key) == 1) {
      continue;
    }

    merged.to_nloc_keys_.emplace_back(nloc_key);
    added_to_nlocs.insert(nloc_key);
  }

  return merged;
}

transfer_result merge(transfer_result const& tres_a,
                      transfer_result const& tres_b) {
  auto merged = transfer_result{};
  auto added_to_nlocs = set<key64_t>{};

  utl::verify(tres_a.from_nloc_key_ == tres_b.from_nloc_key_,
              "Cannot merge two transfer results from different locations.");
  utl::verify(tres_a.profile_ == tres_b.profile_,
              "Cannot merge two transfer results with different profiles.");
  utl::verify(tres_a.to_nloc_keys_.size() == tres_a.infos_.size(),
              "(A) Cannot merge transfer results with invalid target and info "
              "matching.");
  utl::verify(tres_b.to_nloc_keys_.size() == tres_b.infos_.size(),
              "(B) Cannot merge transfer results with invalid target and info "
              "matching.");

  merged.from_nloc_key_ = tres_a.from_nloc_key_;
  merged.profile_ = tres_a.profile_;

  merged.to_nloc_keys_ = tres_a.to_nloc_keys_;
  merged.infos_ = tres_a.infos_;

  // build added_to_nlocs set
  for (auto const& nloc_key : merged.to_nloc_keys_) {
    added_to_nlocs.insert(nloc_key);
  }

  // insert new and unique nloc/info keys
  for (auto const [nloc_key, info] :
       utl::zip(tres_b.to_nloc_keys_, tres_b.infos_)) {
    if (added_to_nlocs.count(nloc_key) == 1) {
      continue;
    }

    merged.to_nloc_keys_.emplace_back(nloc_key);
    merged.infos_.emplace_back(info);
    added_to_nlocs.insert(nloc_key);
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
  tres_repr << "[transfer result] " << to_key(tres) << ": #results - "
            << tres.infos_.size();
  return out << tres_repr.str();
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

}  // namespace motis::footpaths