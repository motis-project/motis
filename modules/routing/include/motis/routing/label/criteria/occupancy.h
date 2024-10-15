#pragma once

#include "motis/core/schedule/edges.h"

namespace motis::routing {

struct occupancy {
  uint16_t occ_;
  uint8_t max_occ_;
  bool sitting_;
};

struct occupancy_initializer {
  template <typename Label, typename LowerBounds>
  static void init(Label& l, LowerBounds&) {
    l.occ_ = 0;
    l.max_occ_ = 0;
    l.sitting_ = false;
  }
};

template <bool Sit>
struct occupancy_updater {
  template <typename Label, typename LowerBounds>
  static void update(Label& l, edge_cost const& ec, LowerBounds&) {
    if (ec.transfer_) {
      l.sitting_ = false;
    }
    if (ec.connection_ != nullptr && (!l.sitting_ || !Sit)) {
      if (ec.connection_->occupancy_ == 0) {
        l.sitting_ = true;
      } else {
        l.occ_ += (ec.connection_->occupancy_ * ec.connection_->travel_time());
        l.max_occ_ = std::max(ec.connection_->occupancy_, l.max_occ_);
      }
    }
  }
};

struct occupancy_dominance {
  template <typename Label>
  struct domination_info {
    domination_info(Label const& a, Label const& b)
        : greater_(a.occ_ > b.occ_), smaller_(a.occ_ < b.occ_) {}
    inline bool greater() const { return greater_; }
    inline bool smaller() const { return smaller_; }
    bool greater_, smaller_;
  };

  template <typename Label>
  static domination_info<Label> dominates(Label const& a, Label const& b) {
    return domination_info<Label>(a, b);
  }
};

struct occupancy_dominance_max {
  template <typename Label>
  struct domination_info {
    domination_info(Label const& a, Label const& b)
        : greater_(a.max_occ_ > b.max_occ_),
          smaller_(a.max_occ_ < b.max_occ_) {}
    inline bool greater() const { return greater_; }
    inline bool smaller() const { return smaller_; }
    bool greater_, smaller_;
  };

  template <typename Label>
  static domination_info<Label> dominates(Label const& a, Label const& b) {
    return domination_info<Label>(a, b);
  }
};

}