#pragma once

#include "motis/core/schedule/edges.h"

namespace motis::routing {

constexpr duration MAX_TRANSFERS = 7;

struct transfers {
  uint8_t transfers_, transfers_lb_;
};

struct transfers_initializer {
  template <typename Label, typename LowerBounds>
  static void init(Label& l, LowerBounds& lb) {
    l.transfers_ = 0;

    auto const lb_val = lb.transfers_[l.get_node()];
    if (lb.transfers_.is_reachable(lb_val)) {
      l.transfers_lb_ = lb_val;
    } else {
      l.transfers_lb_ = std::numeric_limits<uint8_t>::max();
    }
  }
};

struct transfers_updater {
  template <typename Label, typename LowerBounds>
  static void update(Label& l, edge_cost const& ec, LowerBounds& lb) {
    if (ec.transfer_) {
      ++l.transfers_;
    }

    auto const lb_val = lb.transfers_[l.get_node()];
    if (lb.transfers_.is_reachable(lb_val)) {
      l.transfers_lb_ = l.transfers_ + lb_val;
    } else {
      l.transfers_lb_ = std::numeric_limits<uint8_t>::max();
    }
  }
};

struct transfers_dominance {
  template <typename Label>
  struct domination_info {
    domination_info(Label const& a, Label const& b)
        : greater_(a.transfers_lb_ > b.transfers_lb_),
          smaller_(a.transfers_lb_ < b.transfers_lb_) {}
    inline bool greater() const { return greater_; }
    inline bool smaller() const { return smaller_; }
    bool greater_, smaller_;
  };

  template <typename Label>
  static domination_info<Label> dominates(Label const& a, Label const& b) {
    return domination_info<Label>(a, b);
  }
};

struct transfers_filter {
  template <typename Label>
  static bool is_filtered(Label const& l, duration const /* fastest_direct */) {
    return l.transfers_lb_ > MAX_TRANSFERS;
  }
};

}  // namespace motis::routing
