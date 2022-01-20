#pragma once

#include "motis/routing/label/criteria/transfers.h"
#include "motis/routing/label/criteria/travel_time.h"

namespace motis::routing {

constexpr auto TRANSFER_COST = 20;
constexpr auto MAX_WEIGHTED = MAX_TRAVEL_TIME + TRANSFER_COST * MAX_TRANSFERS;

struct weighted {
  duration weighted_, weighted_lb_;
};

struct get_weighted_lb {
  template <typename Label>
  duration operator()(Label const* l) {
    return l->weighted_lb_;
  }
};

struct weighted_initializer {
  template <typename Label, typename LowerBounds>
  static void init(Label& l, LowerBounds& lb) {
    l.weighted_ = std::abs(l.now_ - l.start_);

    auto const tt_lb = lb.travel_time_[l.get_node()];
    auto const ic_lb = lb.transfers_[l.get_node()];
    if (lb.travel_time_.is_reachable(tt_lb) &&
        lb.transfers_.is_reachable(ic_lb)) {
      l.weighted_lb_ = l.weighted_ + tt_lb + (TRANSFER_COST * ic_lb);
    } else {
      l.weighted_lb_ = std::numeric_limits<duration>::max();
    }
  }
};

struct weighted_updater {
  template <typename Label, typename LowerBounds>
  static void update(Label& l, edge_cost const& ec, LowerBounds& lb) {
    l.weighted_ += ec.time_;
    if (ec.transfer_) {
      l.weighted_ += TRANSFER_COST;
    }

    auto const tt_lb = lb.travel_time_[l.get_node()];
    auto const ic_lb = lb.transfers_[l.get_node()];
    if (lb.travel_time_.is_reachable(tt_lb) &&
        lb.transfers_.is_reachable(ic_lb)) {
      l.weighted_lb_ = l.weighted_ + tt_lb + (TRANSFER_COST * ic_lb);
    } else {
      l.weighted_lb_ = std::numeric_limits<duration>::max();
    }
  }
};

struct weighted_dominance {
  template <typename Label>
  struct domination_info {
    domination_info(Label const& a, Label const& b)
        : greater_(a.weighted_lb_ > b.weighted_lb_),
          smaller_(a.weighted_lb_ < b.weighted_lb_) {}
    inline bool greater() const { return greater_; }
    inline bool smaller() const { return smaller_; }
    bool greater_, smaller_;
  };

  template <typename Label>
  static domination_info<Label> dominates(Label const& a, Label const& b) {
    return domination_info<Label>(a, b);
  }
};

struct weighted_filter {
  template <typename Label>
  static bool is_filtered(Label const& l) {
    return l.weighted_lb_ > MAX_WEIGHTED;
  }
};

}  // namespace motis::routing
