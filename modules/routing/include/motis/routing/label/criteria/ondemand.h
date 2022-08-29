#pragma once

#include <cmath>
#include <cstdint>
#include "transfers.h"
#include "travel_time.h"

namespace motis::routing {

constexpr auto TRANSFER_C = 20;
constexpr auto MAX_WEIGHTED_OD = MAX_TRAVEL_TIME + TRANSFER_C * MAX_TRANSFERS;

struct ondemand {
  duration ondemand_, ondemand_lb_;
  bool temp;
};

struct get_ondemand_lb {
  template <typename Label>
  duration operator()(Label const* l) {
    return l->ondemand_lb_;
  }
};

struct ondemand_initializer {
  template <typename Label, typename LowerBounds>
  static void init(Label& l, LowerBounds& lb) {
    l.ondemand_ = std::abs(l.now_ - l.start_);
    l.temp = false;

    auto const tt_lb = lb.travel_time_[l.get_node()];
    auto const t_lb = lb.transfers_[l.get_node()];
    if (lb.travel_time_.is_reachable(tt_lb) && lb.transfers_.is_reachable(t_lb))
    {
      l.ondemand_lb_ = l.ondemand_ + tt_lb + (TRANSFER_C * t_lb);
    }
    else {
      l.ondemand_lb_ = std::numeric_limits<duration>::max();
    }
  }
};

struct ondemand_updater {
  template <typename Label, typename LowerBounds>
  static void update(Label& l, edge_cost const& ec, LowerBounds& lb) {
    l.ondemand_ += ec.time_;
    if (ec.transfer_) {
      l.ondemand_ += TRANSFER_C;
    }
    if(l.edge_->type() == edge::MUMO_EDGE)
    {
      bool od = edge::get_is_ondemand(l.edge_);
      if(od)
      {
        l.temp = true;
      }
    }
    auto const tt_lb = lb.travel_time_[l.get_node()];
    auto const t_lb = lb.transfers_[l.get_node()];
    if (lb.travel_time_.is_reachable(tt_lb) && lb.transfers_.is_reachable(t_lb)) {
      l.ondemand_lb_ = l.ondemand_ + tt_lb + (TRANSFER_C * t_lb);
    }
    else {
      l.ondemand_lb_ = std::numeric_limits<duration>::max();
    }
  }
};

// a od     > b od    -> g
// a od     < b od    -> k
// a od     > b nod   -> k
// a od     < b nod   -> k
// a nod    > b nod   -> g
// a nod    < b nod   -> k
// a nod    > b od    -> g
// a nod    < b od    -> g

// a nicht größer oder a kleiner -> besser

struct ondemand_dominance {
  template <typename Label>
  struct domination_info {
    domination_info(Label const& a, Label const& b)
        : greater_((a.ondemand_lb_ > b.ondemand_lb_ && a.temp == b.temp)
                   || (a.temp && !b.temp)),
          smaller_((a.ondemand_lb_ < b.ondemand_lb_ && a.temp == b.temp)
                   || (!a.temp && b.temp)){}
    inline bool greater() const { return greater_; }
    inline bool smaller() const { return smaller_; }
    bool greater_, smaller_;
  };

  template <typename Label>
  static domination_info<Label> dominates(Label const& a, Label const& b) {
    return domination_info<Label>(a, b);
  }
};

struct ondemand_filter {
  template <typename Label>
  static bool is_filtered(Label const& l) {
    return l.ondemand_lb_ > MAX_WEIGHTED_OD;
  }
};

}   // namespace motis::routing