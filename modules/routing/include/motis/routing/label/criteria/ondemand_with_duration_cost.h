#pragma once

#include <cmath>
#include <cstdint>

namespace motis::routing {

constexpr auto ONDEMAND_COST = 20;

struct ondemand_with_duration_cost {
  bool is_ondemand_;
  duration edge_duration_;
};

struct ondemand_with_duration_cost_initializer {
  template <typename Label, typename LowerBounds>
  static void init(Label& l, LowerBounds&) {
    l.is_ondemand_ = false;
    l.edge_duration_ = 0;
  }
};

struct ondemand_with_duration_cost_updater {
  template <typename Label, typename LowerBounds>
  static void update(Label& l, edge_cost const& ec, LowerBounds&) {
    l.is_ondemand_ = edge::get_is_ondemand(l.edge_);
    if(l.is_ondemand_) {
      l.edge_duration_ += (ec.time_ / 1.5) * ONDEMAND_COST;
    } else {
      l.edge_duration_ += ec.time_;
    }
  }
};

struct ondemand_with_duration_cost_dominance {
  template <typename Label>
  struct domination_info {
    domination_info(Label const& a, Label const& b)
        : greater_(a.edge_duration_ > b.edge_duration_ && (a.is_ondemand_ > b.is_ondemand_)),
          smaller_(a.edge_duration_ < b.edge_duration_ && (a.is_ondemand_ < b.is_ondemand_)){}
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