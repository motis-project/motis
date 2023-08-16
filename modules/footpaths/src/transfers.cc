#include "motis/footpaths/transfers.h"

#include <sstream>
#include "fmt/core.h"

#include "motis/footpaths/database.h"

namespace motis::footpaths {

std::ostream& operator<<(std::ostream& out, transfer_info const& tinfo) {
  auto tinfo_repr =
      fmt::format("dur: {}, dist: {}", tinfo.duration_, tinfo.distance_);
  return out << tinfo_repr;
}

std::ostream& operator<<(std::ostream& out, transfer_result const& tres) {
  std::stringstream tres_repr;
  tres_repr << to_key(tres) << ": " << tres.info_;
  return out << tres_repr.str();
}

std::ostream& operator<<(std::ostream& out, transfer_request const& treq) {
  auto treq_repr = fmt::format("[transfer request] {} has {} targets.",
                               to_key(treq), treq.to_nloc_keys_.size());
  return out << treq_repr;
}

std::ostream& operator<<(std::ostream& out,
                         transfer_request_keys const treq_k) {
  auto treq_k_repr = fmt::format("[transfer request keys] {} has {} targets.",
                                 to_key(treq_k), treq_k.to_nloc_keys_.size());
  return out << treq_k_repr;
}

}  // namespace motis::footpaths