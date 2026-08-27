#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "nigiri/routing/query.h"
#include "nigiri/types.h"

#include "motis/config.h"
#include "motis/fwd.h"

namespace motis {

struct feed_labels {
  using src_set = nigiri::bitvec_map<nigiri::source_idx_t>;

  feed_labels() = default;
  feed_labels(config::timetable const&, tag_lookup const&);

  nigiri::routing::blocked_feeds blocked(
      nigiri::timetable const&,
      nigiri::rt_timetable const*,
      std::vector<std::string> const& include,
      std::vector<std::string> const& exclude) const;

  src_set resolve(std::vector<std::string> const&) const;

  std::unordered_map<std::string, src_set> by_name_;
  cista::base_t<nigiri::source_idx_t> n_srcs_{0U};
};

}  // namespace motis
