#pragma once

namespace motis::routing {

struct default_tb {
  template <typename Label>
  static bool dominates(bool, Label const&, Label const&) {
    return true;
  }
};

struct post_search_tb {
  template <typename Label>
  static bool dominates(bool could_dominate, Label const&, Label const&) {
    return could_dominate;
  }
};

struct absurdity_tb {
  template <typename Label>
  static bool dominates(bool, Label const& a, Label const& b) {
    return a.absurdity_ <= b.absurdity_;
  }
};

struct absurdity_post_search_tb {
  template <typename Label>
  static bool dominates(bool could_dominate, Label const& a, Label const& b) {
    return could_dominate || a.absurdity_ <= b.absurdity_;
  }
};

}  // namespace motis::routing