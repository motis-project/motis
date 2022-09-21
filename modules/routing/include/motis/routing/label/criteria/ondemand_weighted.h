#pragma once

#include <cmath>
#include <cstdint>
#include "transfers.h"
#include "travel_time.h"

namespace motis::routing {

constexpr auto TRANSFER_C = 20;
constexpr auto MAX_WEIGHTED_OD = MAX_TRAVEL_TIME + TRANSFER_C * MAX_TRANSFERS;

struct ondemand_weighted {
  duration ondemand_weighted_, ondemand_weighted_lb_;
  bool is_ondemand_;
};

struct get_ondemand_weighted_lb {
  template <typename Label>
  duration operator()(Label const* l) {
    return l->ondemand_weighted_lb_;
  }
};

struct ondemand_weighted_initializer {
  template <typename Label, typename LowerBounds>
  static void init(Label& l, LowerBounds& lb) {
    l.ondemand_weighted_ = std::abs(l.now_ - l.start_);
    l.is_ondemand_ = false;

    auto const tt_lb = lb.travel_time_[l.get_node()];
    auto const t_lb = lb.transfers_[l.get_node()];
    if (lb.travel_time_.is_reachable(tt_lb) && lb.transfers_.is_reachable(t_lb)) {
      l.ondemand_weighted_lb_ = l.ondemand_weighted_ + tt_lb + (TRANSFER_C * t_lb);
    }
    else {
      l.ondemand_weighted_lb_ = std::numeric_limits<duration>::max();
    }
  }
};

struct ondemand_weighted_updater {
  template <typename Label, typename LowerBounds>
  static void update(Label& l, edge_cost const& ec, LowerBounds& lb) {
    l.ondemand_weighted_ += ec.time_;
    if (ec.transfer_) {
      l.ondemand_weighted_ += TRANSFER_C;
    }
    l.is_ondemand_ = edge::get_is_ondemand(l.edge_);

    auto const tt_lb = lb.travel_time_[l.get_node()];
    auto const t_lb = lb.transfers_[l.get_node()];
    if (lb.travel_time_.is_reachable(tt_lb) && lb.transfers_.is_reachable(t_lb)) {
      l.ondemand_weighted_lb_ = l.ondemand_weighted_ + tt_lb + (TRANSFER_C * t_lb);
    }
    else {
      l.ondemand_weighted_lb_ = std::numeric_limits<duration>::max();
    }
  }
};

struct ondemand_weighted_dominance {
  template <typename Label>
  struct domination_info {
    domination_info(Label const& a, Label const& b)
        : greater_((a.ondemand_weighted_lb_ > b.ondemand_weighted_lb_ && a.is_ondemand_ == b.is_ondemand_)
                   || (a.is_ondemand_ && !b.is_ondemand_)),
          smaller_((a.ondemand_weighted_lb_ < b.ondemand_weighted_lb_ && a.is_ondemand_ == b.is_ondemand_)
                   || (!a.is_ondemand_ && b.is_ondemand_)){}
    inline bool greater() const { return greater_; }
    inline bool smaller() const { return smaller_; }
    bool greater_, smaller_;
  };

  template <typename Label>
  static domination_info<Label> dominates(Label const& a, Label const& b) {
    return domination_info<Label>(a, b);
  }
};

struct ondemand_weighted_filter {
  template <typename Label>
  static bool is_filtered(Label const& l) {
    return l.ondemand_weighted_lb_ > MAX_WEIGHTED_OD;
  }
};

}   // namespace motis::routing