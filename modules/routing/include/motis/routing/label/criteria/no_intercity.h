#pragma once

namespace motis::routing {

struct no_intercity_filter {
  template <typename Label>
  static bool is_filtered(Label const& l) {
    if (l.connection_ != nullptr) {
      return l.connection_->full_con_->clasz_ < service_class::RE;
    } else {
      return false;
    }
  }
};

}  // namespace motis::routing
