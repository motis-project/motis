#pragma once

#include <cmath>
#include <cstdint>

#include "motis/core/schedule/edges.h"

namespace motis::routing {

struct accessibility {
  uint16_t accessibility_;
};

struct accessibility_initializer {
  template <typename Label, typename LowerBounds>
  static void init(Label& l, LowerBounds&) {
    l.accessibility_ = 0;
  }
};

struct accessibility_updater {
  template <typename Label, typename LowerBounds>
  static void update(Label& l, edge_cost const& ec, LowerBounds&) {
    unsigned val = static_cast<unsigned>(l.accessibility_) +
                   static_cast<unsigned>(ec.accessibility_);
    l.accessibility_ = static_cast<uint16_t>(std::min(
        val, static_cast<unsigned>(std::numeric_limits<uint16_t>::max())));
  }
};

struct accessibility_dominance {
  template <typename Label>
  struct domination_info {
    domination_info(Label const& a, Label const& b)
        : greater_(a.accessibility_ > b.accessibility_),
          smaller_(a.accessibility_ < b.accessibility_) {}
    inline bool greater() const { return greater_; }
    inline bool smaller() const { return smaller_; }
    bool greater_, smaller_;
  };

  template <typename Label>
  static domination_info<Label> dominates(Label const& a, Label const& b) {
    return domination_info<Label>(a, b);
  }
};

}  // namespace motis::routing
