#pragma once
#ifdef MOTIS_CAPACITY_IN_SCHEDULE

#include <cmath>
#include <cstdint>

#include "motis/core/schedule/edges.h"

namespace motis::routing {

struct load_sum {
  uint16_t load_sum_;
};

struct load_sum_initializer {
  template <typename Label, typename LowerBounds>
  static void init(Label& l, LowerBounds&) {
    l.load_sum_ = 0;
  }
};

struct load_sum_updater {
  template <typename Label, typename LowerBounds>
  static void update(Label& l, edge_cost const& ec, LowerBounds&) {
    if (ec.transfer_) {
      l.load_sum_ += 30;
    }
    if (ec.capacity_ != 0) {
      auto const load =
          static_cast<float>(ec.passengers_) / static_cast<float>(ec.capacity_);
      if (load > 1.0F) {
        l.load_sum_ += ec.time_;
      } else if (load > 0.65F) {
        l.load_sum_ += static_cast<uint16_t>(ec.time_ * 0.2F);
      }
    }
  }
};

struct load_sum_dominance {
  template <typename Label>
  struct domination_info {
    domination_info(Label const& a, Label const& b)
        : greater_(a.load_sum_ > b.load_sum_),
          smaller_(a.load_sum_ < b.load_sum_) {}
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

#endif
