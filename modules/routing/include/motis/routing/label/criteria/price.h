#pragma once

#include "motis/core/schedule/edges.h"

namespace motis::routing {

constexpr auto const MAX_REGIONAL_TRAIN_TICKET_PRICE = 4200U;
constexpr auto const MINUTELY_WAGE = 8;

struct price {
  uint16_t regional_price_;
  uint16_t other_price_;
  uint16_t total_price_;
};

template <typename T>
inline uint16_t add_price(
    uint16_t base, T additional,
    uint16_t limit = std::numeric_limits<uint16_t>::max()) {
  return static_cast<uint16_t>(std::min(
      static_cast<uint32_t>(limit),
      static_cast<uint32_t>(base) + static_cast<uint32_t>(additional)));
}

struct price_initializer {
  template <typename Label, typename LowerBounds>
  static void init(Label& l, LowerBounds&) {
    l.regional_price_ = 0;
    l.other_price_ = 0;
    l.total_price_ = 0;
  }
};

struct price_updater {
  template <typename Label, typename LowerBounds>
  static void update(Label& l, edge_cost const& ec, LowerBounds&) {
    if (ec.connection_ != nullptr) {
      connection const* con = ec.connection_->full_con_;
      if (con->clasz_ >= service_class::RB) {
        l.regional_price_ = add_price(l.regional_price_, con->price_,
                                      MAX_REGIONAL_TRAIN_TICKET_PRICE);
      } else {
        l.other_price_ = add_price(l.other_price_, con->price_);
      }
    } else {
      l.other_price_ = add_price(l.other_price_, ec.price_);
    }
    l.total_price_ = add_price(
        l.other_price_, l.regional_price_ + l.travel_time_ * MINUTELY_WAGE);
  }
};

struct price_dominance {
  template <typename Label>
  struct domination_info {
    domination_info(Label const& a, Label const& b) {
      uint32_t const min_wage_diff = a.now_ > b.now_
                                         ? (a.now_ - b.now_) * MINUTELY_WAGE
                                         : (b.now_ - a.now_) * MINUTELY_WAGE;
      auto const fwd = Label::dir == search_dir::FWD;

      auto const regional_remaining_a =
          MAX_REGIONAL_TRAIN_TICKET_PRICE - a.regional_price_;
      auto const regional_remaining_b =
          MAX_REGIONAL_TRAIN_TICKET_PRICE - b.regional_price_;

      uint32_t const a_price =
          (a.now_ > b.now_) == fwd
              ? a.total_price_ + regional_remaining_b
              : a.total_price_ + regional_remaining_b + min_wage_diff;
      uint32_t const b_price =
          (a.now_ > b.now_) == fwd
              ? b.total_price_ + regional_remaining_a + min_wage_diff
              : b.total_price_ + regional_remaining_a;

      greater_ = a_price > b_price;
      smaller_ = a_price < b_price;
    }
    inline bool greater() const { return greater_; }
    inline bool smaller() const { return smaller_; }

    bool greater_, smaller_;
  };

  template <typename Label>
  static domination_info<Label> dominates(Label const& a, Label const& b) {
    auto dom_info = domination_info<Label>(a, b);
    return dom_info;
  }
};

}  // namespace motis::routing
