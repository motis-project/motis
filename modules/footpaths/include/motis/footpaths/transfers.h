#pragma once

#include <ostream>
#include <string>
#include <vector>

#include "fmt/core.h"

#include "motis/footpaths/platforms.h"
#include "motis/footpaths/types.h"

#include "nigiri/types.h"

namespace motis::footpaths {

struct transfer_request_keys {
  string from_pf_key_;
  string from_nloc_key_;

  strings to_pf_keys_;
  strings to_nloc_keys_;

  string profile_;
};
using transfer_requests_keys = std::vector<transfer_request_keys>;

struct transfer_request {
  platform transfer_start_;
  string from_nloc_key_;

  platforms transfer_targets_;
  std::vector<string> to_nloc_keys_;

  std::string profile_;
};
using transfer_requests = std::vector<transfer_request>;

struct transfer_info {
  nigiri::duration_t duration_{};
  double distance_{};
};
using transfer_infos = std::vector<transfer_info>;

struct transfer_result {
  string from_nloc_key_;
  string to_nloc_key_;
  string profile_;

  transfer_info info_;
};
using transfer_results = std::vector<transfer_result>;

// keys :: ids
inline string to_key(transfer_request_keys const& treq_k) {
  return {fmt::format("{}::{}", treq_k.from_nloc_key_, treq_k.profile_)};
}

inline string to_key(transfer_request const& treq) {
  return {fmt::format("{}::{}", treq.from_nloc_key_, treq.profile_)};
}

inline string to_key(transfer_result const& tr) {
  return {fmt::format("{}::{}::{}", tr.from_nloc_key_, tr.to_nloc_key_,
                      tr.profile_)};
}

// ostreams
std::ostream& operator<<(std::ostream&, transfer_info const&);
std::ostream& operator<<(std::ostream&, transfer_result const&);
std::ostream& operator<<(std::ostream&, transfer_request const&);
std::ostream& operator<<(std::ostream&, transfer_request_keys const&);

}  // namespace motis::footpaths
