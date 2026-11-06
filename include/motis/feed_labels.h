#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "nigiri/routing/query.h"
#include "nigiri/types.h"

#include "motis/adr_extend_tt.h"
#include "motis/config.h"
#include "motis/fwd.h"

namespace motis {

struct feed_labels {
  using src_set = nigiri::bitvec_map<nigiri::source_idx_t>;

  feed_labels() = default;
  feed_labels(config::timetable const&, tag_lookup const&);

  void add_places(nigiri::timetable const&, adr_ext const&);

  src_set active(std::vector<std::string> const& include,
                 std::vector<std::string> const& exclude) const;

  nigiri::routing::blocked_feeds blocked(
      nigiri::timetable const&,
      nigiri::rt_timetable const*,
      std::vector<std::string> const& include,
      std::vector<std::string> const& exclude) const;

  bool is_active(src_set const& active, adr_extra_place_idx_t) const;

  src_set resolve(std::vector<std::string> const&) const;

  std::unordered_map<std::string, src_set> by_name_;
  src_set hidden_;
  cista::base_t<nigiri::source_idx_t> n_srcs_{0U};
  nigiri::vecvec<adr_extra_place_idx_t, nigiri::source_idx_t> place_srcs_;
};

}  // namespace motis
