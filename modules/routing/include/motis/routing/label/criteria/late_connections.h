#pragma once

#include "motis/core/schedule/edges.h"

namespace motis::routing {

auto constexpr NIGHT_BEGIN = 1380; /* 23:00 GMT */
auto constexpr NIGHT_END = 240; /* 04:00 GMT */

inline uint16_t night_travel_duration(time const travel_begin,
                                      time const travel_end,
                                      uint16_t const night_begin = NIGHT_BEGIN,
                                      uint16_t const night_end = NIGHT_END) {
  auto intersection = [](uint16_t const begin1, uint16_t const end1,
                         uint16_t const begin2, uint16_t const end2) {
    assert(begin1 < end1);
    assert(begin2 < end2);
    assert(begin1 < MINUTES_A_DAY);
    assert(end1 < MINUTES_A_DAY);
    assert(begin2 < MINUTES_A_DAY);
    assert(end2 < MINUTES_A_DAY);
    auto const b = std::max(begin1, begin2);
    auto const e = std::min(end1, end2);
    return static_cast<uint16_t>(std::max(0, e - b));
  };

  if (travel_begin >= travel_end || night_begin == night_end) {
    return 0;
  }
  if (travel_end - travel_begin > MINUTES_A_DAY) {
    auto e = NIGHT_END;
    if (NIGHT_END < NIGHT_BEGIN) {
      e += MINUTES_A_DAY;
    }
    return (e - NIGHT_BEGIN) +
           night_travel_duration(travel_begin + MINUTES_A_DAY, travel_end,
                                 night_begin, night_end);
  }

  assert(night_begin < MINUTES_A_DAY);
  assert(night_end < MINUTES_A_DAY);

  uint16_t const tb = travel_begin % MINUTES_A_DAY;
  uint16_t const te = travel_end % MINUTES_A_DAY;

  if (night_begin < night_end) {
    if (tb < te) {
      return intersection(tb, te, night_begin, night_end);
    } else if (te > night_begin) {
      return te - night_begin;
    }
  } else if (tb < te) {
    return (te - tb) - intersection(tb, te, night_end, night_begin);
  } else {
    auto const offset = MINUTES_A_DAY - std::min(night_begin, tb);
    return intersection((tb + offset) % MINUTES_A_DAY,
                        (te + offset) % MINUTES_A_DAY,
                        (night_begin + offset) % MINUTES_A_DAY,
                        (night_end + offset) % MINUTES_A_DAY);
  }

  return 0;
}

struct late_connections {
  uint16_t night_penalty_, db_costs_;
  uint8_t visited_hotel_;
  enum hotel { NOT_VISITED, VISITED, FILTERED };
};

struct late_connections_initializer {
  template <typename Label, typename LowerBounds>
  static void init(Label& l, LowerBounds&) {
    l.night_penalty_ = 0;
    l.db_costs_ = 0;
    l.visited_hotel_ = late_connections::NOT_VISITED;
  }
};

struct late_connections_updater {
  template <typename Label, typename LowerBounds>
  static void update(Label& l, edge_cost const& ec, LowerBounds&) {
    l.db_costs_ += ec.price_;
    if (l.edge_->type() == edge::HOTEL_EDGE) {
      if (l.visited_hotel_ == late_connections::NOT_VISITED) {
        l.visited_hotel_ = late_connections::VISITED;
      }
    } else if (l.edge_->type() == edge::PERIODIC_MUMO_EDGE /* taxi */ &&
               l.visited_hotel_ == late_connections::VISITED) {
      /* taxi after hotel not allowed */
      l.visited_hotel_ = late_connections::FILTERED;
    } else {
      l.night_penalty_ += night_travel_duration(l.now_ - ec.time_, l.now_);
    }
  }
};

struct late_connections_dominance {
  template <typename Label>
  struct domination_info {
    domination_info(Label const& a, Label const& b)
        : greater_(a.visited_hotel_ != b.visited_hotel_ ||
                   a.db_costs_ > b.db_costs_ ||
                   a.night_penalty_ > b.night_penalty_),
          smaller_(a.visited_hotel_ == b.visited_hotel_ &&
                   a.db_costs_ <= b.db_costs_ &&
                   a.night_penalty_ <= b.night_penalty_ &&
                   (a.db_costs_ < b.db_costs_ ||
                    a.night_penalty_ < b.night_penalty_)) {}
    inline bool greater() const { return greater_; }
    inline bool smaller() const { return smaller_; }
    bool greater_, smaller_;
  };

  template <typename Label>
  static domination_info<Label> dominates(Label const& a, Label const& b) {
    return domination_info<Label>(a, b);
  }
};

template <unsigned DBCostRelaxation>
struct late_connections_post_search_dominance_base {
  template <typename Label>
  struct domination_info {
    domination_info(Label const& a, Label const& b)
        : greater_(a.db_costs_ > b.db_costs_ + DBCostRelaxation ||
                   a.night_penalty_ > b.night_penalty_),
          smaller_(a.db_costs_ + DBCostRelaxation <= b.db_costs_ &&
                   a.night_penalty_ <= b.night_penalty_ &&
                   (a.db_costs_ + DBCostRelaxation < b.db_costs_ ||
                    a.night_penalty_ < b.night_penalty_)) {}
    inline bool greater() const { return greater_; }
    inline bool smaller() const { return smaller_; }
    bool greater_, smaller_;
  };

  template <typename Label>
  static domination_info<Label> dominates(Label const& a, Label const& b) {
    return domination_info<Label>(a, b);
  }
};

using late_connections_post_search_dominance =
    late_connections_post_search_dominance_base<1000>;
using late_connections_post_search_dominance_for_tests =
    late_connections_post_search_dominance_base<0>;

struct late_connections_filter {
  template <typename Label>
  static bool is_filtered(Label const& l) {
    return l.visited_hotel_ == late_connections::FILTERED;
  }
};

}  // namespace motis::routing
