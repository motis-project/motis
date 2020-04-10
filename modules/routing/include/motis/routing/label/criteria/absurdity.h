#pragma once

#include "motis/core/schedule/edges.h"

namespace motis::routing {

constexpr auto MAX_SUCCESSIVE_FOOT_EDGES_ALLOWED = 3;

struct absurdity {
  uint8_t absurdity_, foot_counter_;
};

struct absurdity_initializer {
  template <typename Label, typename LowerBounds>
  static void init(Label& l, LowerBounds&) {
    l.absurdity_ = 0;
    l.foot_counter_ = 0;
  }
};

struct absurdity_updater {
  template <typename Label, typename LowerBounds>
  static void update(Label& l, edge_cost const&, LowerBounds&) {
    if (l.edge_->type() == edge::FOOT_EDGE ||
        l.edge_->type() == edge::AFTER_TRAIN_FWD_EDGE ||
        l.edge_->type() == edge::AFTER_TRAIN_BWD_EDGE ||
        l.edge_->type() == edge::ENTER_EDGE ||
        l.edge_->type() == edge::EXIT_EDGE) {
      if (l.foot_counter_ <= MAX_SUCCESSIVE_FOOT_EDGES_ALLOWED) {
        ++l.foot_counter_;
      }
      if (l.foot_counter_ > MAX_SUCCESSIVE_FOOT_EDGES_ALLOWED &&
          l.absurdity_ < UINT8_MAX) {
        ++l.absurdity_;
      }
    } else {
      l.foot_counter_ = 0;
    }
  }
};

}  // namespace motis::routing
