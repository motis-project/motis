#pragma once
#ifdef MOTIS_CAPACITY_IN_SCHEDULE

#include <cmath>
#include <cstdint>

#include "motis/core/schedule/edges.h"

namespace motis::routing {

struct perceived_travel_time {
  uint16_t perceived_travel_time_;
};

struct perceived_travel_time_initializer {
  template <typename Label, typename LowerBounds>
  static void init(Label& l, LowerBounds&) {
    l.perceived_travel_time_ = 0;
  }
};

struct perceived_travel_time_updater {
  template <typename Label, typename LowerBounds>
  static void update(Label& l, edge_cost const& ec, LowerBounds&) {
    l.perceived_travel_time_ += ec.time_;
    if (ec.transfer_) {
      l.perceived_travel_time_ += 30;
    }
    if (ec.capacity_ != 0) {
      auto const load =
          static_cast<float>(ec.passengers_) / static_cast<float>(ec.capacity_);
      if (load > 1.0F) {
        l.perceived_travel_time_ += ec.time_;
      } else if (load > 0.65F) {
        l.perceived_travel_time_ += static_cast<uint16_t>(ec.time_ * 0.2F);
      }
    }
  }
};

struct perceived_travel_time_dominance {
  template <typename Label>
  struct domination_info {
    domination_info(Label const& a, Label const& b)
        : greater_(a.perceived_travel_time_ > b.perceived_travel_time_),
          smaller_(a.perceived_travel_time_ < b.perceived_travel_time_) {}
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
