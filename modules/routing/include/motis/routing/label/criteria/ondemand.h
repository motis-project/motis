#pragma once

#include <cmath>
#include <cstdint>

namespace motis::routing {

struct ondemand {
  bool is_ondemand_;
};

struct ondemand_initializer {
  template <typename Label, typename LowerBounds>
  static void init(Label& l, LowerBounds& lb) {
    l.is_ondemand_ = false;
  }
};

struct ondemand_updater {
  template <typename Label, typename LowerBounds>
  static void update(Label& l, edge_cost const& ec, LowerBounds& lb) {
    l.is_ondemand_ = edge::get_is_ondemand(l.edge_);
  }
};

struct ondemand_dominance {
  template <typename Label>
  struct domination_info {
    domination_info(Label const& a, Label const& b)
        : greater_(a.is_ondemand_ > b.is_ondemand_),
          smaller_(a.is_ondemand_ < b.is_ondemand_){}
    inline bool greater() const { return greater_; }
    inline bool smaller() const { return smaller_; }
    bool greater_, smaller_;
  };

  template <typename Label>
  static domination_info<Label> dominates(Label const& a, Label const& b) {
    return domination_info<Label>(a, b);
  }
};

}   // namespace motis::routing