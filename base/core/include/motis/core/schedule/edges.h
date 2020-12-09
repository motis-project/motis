#pragma once

#include <cassert>
#include <algorithm>

#include "motis/vector.h"

#include "motis/core/common/constants.h"
#include "motis/core/schedule/connection.h"
#include "motis/core/schedule/time.h"

namespace motis {

struct node;

struct edge_cost {
  edge_cost()
      : connection_(nullptr),
        time_(INVALID_TIME),
        price_(0),
        transfer_(false),
        accessibility_(0) {}

  edge_cost(duration time, light_connection const* c)
      : connection_(c),
        time_(time),
        price_(0),
        transfer_(false),
        accessibility_(0) {
#ifdef MOTIS_CAPACITY_IN_SCHEDULE
    capacity_ = c->capacity_;
    passengers_ = c->passengers_;
#endif
  }

  explicit edge_cost(duration time, bool transfer = false, uint16_t price = 0,
                     uint16_t accessibility = 0)
      : connection_(nullptr),
        time_(time),
        price_(price),
        transfer_(transfer),
        accessibility_(accessibility) {}

  bool is_valid() const { return time_ != INVALID_TIME; }

  light_connection const* connection_;
  duration time_;
  uint16_t price_;
  bool transfer_;
  uint16_t accessibility_;
#ifdef MOTIS_CAPACITY_IN_SCHEDULE
  uint16_t capacity_{};
  uint16_t passengers_{};
#endif
};

const edge_cost NO_EDGE = edge_cost();

enum class search_dir { FWD, BWD };

class edge {
public:
  enum type {
    INVALID_EDGE,
    ROUTE_EDGE,
    FOOT_EDGE,
    AFTER_TRAIN_FWD_EDGE,
    AFTER_TRAIN_BWD_EDGE,
    MUMO_EDGE,
    PERIODIC_MUMO_EDGE,
    TIME_DEPENDENT_MUMO_EDGE,
    HOTEL_EDGE,
    THROUGH_EDGE,
    ENTER_EDGE,
    EXIT_EDGE,
    FWD_EDGE,
    BWD_EDGE
  };

  edge() : from_{nullptr}, to_{nullptr} {}

  /** route edge constructor. */
  edge(node* from, node* to, mcd::vector<light_connection> const& connections)
      : from_(from), to_(to) {
    m_.type_ = ROUTE_EDGE;
    if (!connections.empty()) {
      m_.route_edge_.init_empty();
      m_.route_edge_.conns_.set(std::begin(connections), std::end(connections));
      std::sort(std::begin(m_.route_edge_.conns_),
                std::end(m_.route_edge_.conns_));
    }
  }

  /** foot edge constructor. */
  edge(node* from, node* to, uint8_t type, uint16_t time_cost, uint16_t price,
       bool transfer, int mumo_id = 0, uint16_t interval_begin = 0,
       uint16_t interval_end = 0, uint16_t accessibility = 0)
      : from_(from), to_(to) {
    m_.type_ = type;
    m_.foot_edge_.time_cost_ = time_cost;
    m_.foot_edge_.price_ = price;
    m_.foot_edge_.transfer_ = transfer;
    m_.foot_edge_.mumo_id_ = mumo_id;
    m_.foot_edge_.interval_begin_ = interval_begin;
    m_.foot_edge_.interval_end_ = interval_end;
    m_.foot_edge_.accessibility_ = accessibility;

    assert(m_.type_ != ROUTE_EDGE);
  }

  /** hotel edge constructor. */
  edge(node* station_node, uint16_t checkout_time, uint16_t min_stay_duration,
       uint16_t price, int mumo_id)
      : from_(station_node), to_(station_node) {
    m_.type_ = HOTEL_EDGE;
    m_.hotel_edge_.checkout_time_ = checkout_time;
    m_.hotel_edge_.min_stay_duration_ = min_stay_duration;
    m_.hotel_edge_.price_ = price;
    m_.hotel_edge_.mumo_id_ = mumo_id;
  }

  template <search_dir Dir = search_dir::FWD>
  edge_cost get_edge_cost(time start_time,
                          light_connection const* last_con) const {
    switch (m_.type_) {
      case ROUTE_EDGE: return get_route_edge_cost<Dir>(start_time);

      case ENTER_EDGE:
        if (Dir == search_dir::FWD) {
          return get_foot_edge_no_cost();
        } else {
          return last_con == nullptr ? NO_EDGE : get_foot_edge_cost();
        }

      case EXIT_EDGE:
        if (Dir == search_dir::FWD) {
          return last_con == nullptr ? NO_EDGE : get_foot_edge_cost();
        } else {
          return get_foot_edge_no_cost();
        }

      case AFTER_TRAIN_FWD_EDGE:
        if (Dir == search_dir::FWD) {
          return last_con == nullptr ? NO_EDGE : get_foot_edge_cost();
        } else {
          return NO_EDGE;
        }

      case AFTER_TRAIN_BWD_EDGE:
        if (Dir == search_dir::BWD) {
          return last_con == nullptr ? NO_EDGE : get_foot_edge_cost();
        } else {
          return NO_EDGE;
        }

      case FWD_EDGE:
        if (Dir == search_dir::FWD) {
          return get_foot_edge_cost();
        } else {
          return NO_EDGE;
        }

      case BWD_EDGE:
        if (Dir == search_dir::BWD) {
          return get_foot_edge_cost();
        } else {
          return NO_EDGE;
        }

      case MUMO_EDGE:
      case FOOT_EDGE: return get_foot_edge_cost();

      case PERIODIC_MUMO_EDGE:
        return edge_cost(calc_time_off(m_.foot_edge_.interval_begin_,
                                       m_.foot_edge_.interval_end_,
                                       start_time % MINUTES_A_DAY) +
                             m_.foot_edge_.time_cost_,
                         m_.foot_edge_.transfer_, m_.foot_edge_.price_);

      case TIME_DEPENDENT_MUMO_EDGE:
        return calc_cost_time_dependent_edge(start_time);

      case HOTEL_EDGE: return calc_cost_hotel_edge(start_time);

      case THROUGH_EDGE:
        return last_con == nullptr ? NO_EDGE : edge_cost(0, false, 0);

      default: return NO_EDGE;
    }
  }

  inline edge_cost get_foot_edge_cost() const {
    return edge_cost(m_.foot_edge_.time_cost_, m_.foot_edge_.transfer_,
                     m_.foot_edge_.price_, m_.foot_edge_.accessibility_);
  }

  static inline edge_cost get_foot_edge_no_cost() {
    return edge_cost(0, false, 0, 0);
  }

  edge_cost get_minimum_cost() const {
    if (m_.type_ == INVALID_EDGE) {
      return NO_EDGE;
    } else if (m_.type_ == ROUTE_EDGE) {
      if (m_.route_edge_.conns_.empty()) {
        return NO_EDGE;
      } else {
        return edge_cost(
            std::min_element(
                std::begin(m_.route_edge_.conns_),
                std::end(m_.route_edge_.conns_),
                [](light_connection const& c1, light_connection const& c2) {
                  return c1.travel_time() < c2.travel_time();
                })
                ->travel_time(),
            false, std::begin(m_.route_edge_.conns_)->full_con_->price_);
      }
    } else if (m_.type_ == FOOT_EDGE || m_.type_ == AFTER_TRAIN_FWD_EDGE ||
               m_.type_ == AFTER_TRAIN_BWD_EDGE || m_.type_ == ENTER_EDGE ||
               m_.type_ == EXIT_EDGE || m_.type_ == BWD_EDGE) {
      return edge_cost(0, m_.foot_edge_.transfer_);
    } else if (m_.type_ == HOTEL_EDGE) {
      return edge_cost(0, false, m_.hotel_edge_.price_);
    } else if (m_.type_ == MUMO_EDGE || m_.type_ == TIME_DEPENDENT_MUMO_EDGE) {
      return edge_cost(0, false, m_.foot_edge_.price_,
                       m_.foot_edge_.accessibility_);
    } else {
      return edge_cost(0);
    }
  }

  inline light_connection const* get_next_valid_lcon(light_connection const* lc,
                                                     unsigned skip = 0) const {
    assert(type() == ROUTE_EDGE);
    assert(lc);

    auto it = lc;
    while (it != end(m_.route_edge_.conns_)) {
      if (skip == 0 && (it->valid_ != 0U)) {
        return it;
      }
      ++it;
      if (skip != 0) {
        --skip;
      }
    }
    return nullptr;
  }

  inline light_connection const* get_prev_valid_lcon(light_connection const* lc,
                                                     unsigned skip = 0) const {
    assert(type() == ROUTE_EDGE);
    assert(lc);

    auto it = std::reverse_iterator<light_connection const*>(lc);
    --it;
    while (it != m_.route_edge_.conns_.rend()) {
      if (skip == 0 && (it->valid_ != 0U)) {
        return &*it;
      }
      ++it;
      if (skip != 0) {
        --skip;
      }
    }
    return nullptr;
  }

  template <search_dir Dir = search_dir::FWD>
  light_connection const* get_connection(time const start_time) const {
    assert(type() == ROUTE_EDGE);

    if (m_.route_edge_.conns_.empty()) {
      return nullptr;
    }

    if (Dir == search_dir::FWD) {
      auto it = std::lower_bound(std::begin(m_.route_edge_.conns_),
                                 std::end(m_.route_edge_.conns_),
                                 light_connection(start_time));

      if (it == std::end(m_.route_edge_.conns_)) {
        return nullptr;
      } else {
        return get_next_valid_lcon(&*it);
      }
    } else {
      auto it = std::lower_bound(
          m_.route_edge_.conns_.rbegin(), m_.route_edge_.conns_.rend(),
          light_connection(0, start_time, nullptr),
          [](light_connection const& lhs, light_connection const& rhs) {
            return lhs.a_time_ > rhs.a_time_;
          });

      if (it == m_.route_edge_.conns_.rend()) {
        return nullptr;
      } else {
        return get_prev_valid_lcon(&*it);
      }
    }
  }

  template <search_dir Dir = search_dir::FWD>
  edge_cost get_route_edge_cost(time const start_time) const {
    assert(type() == ROUTE_EDGE);

    light_connection const* c = get_connection<Dir>(start_time);
    return (c == nullptr)
               ? NO_EDGE
               : edge_cost((Dir == search_dir::FWD) ? c->a_time_ - start_time
                                                    : start_time - c->d_time_,
                           c);
  }

  template <search_dir Dir = search_dir::FWD>
  inline node const* get_destination() const {
    return (Dir == search_dir::FWD) ? to_ : from_;
  }

  template <search_dir Dir = search_dir::FWD>
  inline node const* get_source() const {
    return (Dir == search_dir::FWD) ? from_ : to_;
  }

  inline node const* get_destination(search_dir dir = search_dir::FWD) const {
    return (dir == search_dir::FWD) ? to_ : from_;
  }

  inline node const* get_source(search_dir dir = search_dir::FWD) const {
    return (dir == search_dir::FWD) ? from_ : to_;
  }

  inline bool valid() const { return type() != INVALID_EDGE; }

  inline uint8_t type() const { return m_.type_; }

  inline char const* type_str() const {
    switch (type()) {
      case ROUTE_EDGE: return "ROUTE_EDGE";
      case FOOT_EDGE: return "FOOT_EDGE";
      case AFTER_TRAIN_FWD_EDGE: return "AFTER_TRAIN_FWD_EDGE";
      case AFTER_TRAIN_BWD_EDGE: return "AFTER_TRAIN_BWD_EDGE";
      case MUMO_EDGE: return "MUMO_EDGE";
      case TIME_DEPENDENT_MUMO_EDGE: return "TIME_DEPENDENT_MUMO_EDGE";
      case PERIODIC_MUMO_EDGE: return "PERIODIC_MUMO_EDGE";
      case HOTEL_EDGE: return "HOTEL_EDGE";
      case THROUGH_EDGE: return "THROUGH_EDGE";
      case ENTER_EDGE: return "ENTER_EDGE";
      case EXIT_EDGE: return "EXIT_EDGE";
      case FWD_EDGE: return "FWD_EDGE";
      case BWD_EDGE: return "BWD_EDGE";
      default: return "INVALID";
    }
  }

  int get_mumo_id() const {
    switch (type()) {
      case TIME_DEPENDENT_MUMO_EDGE:
      case PERIODIC_MUMO_EDGE:
      case MUMO_EDGE: return m_.foot_edge_.mumo_id_;
      case HOTEL_EDGE: return m_.hotel_edge_.mumo_id_;
      default: return -1;
    }
  }

  inline bool empty() const {
    return (type() != ROUTE_EDGE) ? true : m_.route_edge_.conns_.empty();
  }

  static time calc_time_off(time const period_begin, time const period_end,
                            time const timestamp) {
    assert(period_begin < MINUTES_A_DAY);
    assert(period_end < MINUTES_A_DAY);
    assert(timestamp < MINUTES_A_DAY);

    /* validity-period begins and ends at the same day */
    if (period_begin <= period_end) {
      if (timestamp < period_begin) {
        return period_begin - timestamp;
      } else if (timestamp > period_end) {
        return (MINUTES_A_DAY - timestamp) + period_begin;
      }
    }
    /* validity-period is over midnight */
    else if (timestamp > period_end && timestamp < period_begin) {
      return period_begin - timestamp;
    }

    return 0;
  }

  ptr<node> from_;
  ptr<node> to_;

  union edge_details {
    edge_details() {  // NOLINT
      std::memset(static_cast<void*>(this), 0, sizeof(*this));  // NOLINT
    }

    edge_details(edge_details&& other) noexcept {  // NOLINT
      type_ = other.type_;
      if (type_ == ROUTE_EDGE) {
        route_edge_.init_empty();
        route_edge_ = std::move(other.route_edge_);
      } else if (type_ == HOTEL_EDGE) {
        hotel_edge_ = other.hotel_edge_;
      } else {
        foot_edge_ = other.foot_edge_;
      }
    }

    edge_details(edge_details const& other) {  // NOLINT
      type_ = other.type_;
      if (type_ == ROUTE_EDGE) {
        route_edge_.init_empty();
        route_edge_ = other.route_edge_;
      } else if (type_ == HOTEL_EDGE) {
        hotel_edge_ = other.hotel_edge_;
      } else {
        foot_edge_.init_empty();
        foot_edge_ = other.foot_edge_;
      }
    }

    edge_details& operator=(edge_details&& other) noexcept {
      type_ = other.type_;
      if (type_ == ROUTE_EDGE) {
        route_edge_.init_empty();
        route_edge_ = std::move(other.route_edge_);
      } else if (type_ == HOTEL_EDGE) {
        hotel_edge_ = other.hotel_edge_;
      } else {
        foot_edge_.init_empty();
        foot_edge_ = other.foot_edge_;
      }

      return *this;
    }

    edge_details& operator=(edge_details const& other) {
      if (this == &other) {
        return *this;
      }

      type_ = other.type_;
      if (type_ == ROUTE_EDGE) {
        route_edge_.init_empty();
        route_edge_ = other.route_edge_;
      } else if (type_ == HOTEL_EDGE) {
        hotel_edge_ = other.hotel_edge_;
      } else {
        foot_edge_.init_empty();
        foot_edge_ = other.foot_edge_;
      }

      return *this;
    }

    ~edge_details() {
      if (type_ == ROUTE_EDGE) {
        using Type = decltype(route_edge_.conns_);
        route_edge_.conns_.~Type();
      }
    }

    // placeholder
    uint8_t type_;

    // TYPE = ROUTE_EDGE
    struct re {
      uint8_t type_padding_;
      mcd::vector<light_connection> conns_;

      void init_empty() { new (&conns_) mcd::vector<light_connection>(); }
    } route_edge_;

    // TYPE = FOOT_EDGE & CO
    struct fe {
      uint8_t type_padding_;

      // edge weight
      uint16_t time_cost_;
      uint16_t price_;
      bool transfer_;

      // id for mumo edge
      int32_t mumo_id_;

      // interval: time-dependent/periodic mumo edge
      uint16_t interval_begin_;
      uint16_t interval_end_;

      uint16_t accessibility_;

      void init_empty() {
        time_cost_ = 0;
        price_ = 0;
        transfer_ = false;
        mumo_id_ = -1;
        accessibility_ = 0;
      }
    } foot_edge_;

    // TYPE = HOTEL_EDGE
    struct he {
      uint8_t type_padding_;
      uint16_t checkout_time_;
      uint16_t min_stay_duration_;
      uint16_t price_;
      uint32_t mumo_id_;
    } hotel_edge_;
  } m_;

private:
  edge_cost calc_cost_time_dependent_edge(time const start_time) const {
    if (start_time > m_.foot_edge_.interval_end_) {
      return NO_EDGE;
    }
    auto const time_off =
        std::max(0, m_.foot_edge_.interval_begin_ - start_time);
    return edge_cost(time_off + m_.foot_edge_.time_cost_,
                     m_.foot_edge_.transfer_, m_.foot_edge_.price_);
  }

  edge_cost calc_cost_hotel_edge(time const start_time) const {
    uint16_t const offset =
        start_time % MINUTES_A_DAY < m_.hotel_edge_.checkout_time_
            ? 0
            : MINUTES_A_DAY;
    auto const duration = std::max(
        m_.hotel_edge_.min_stay_duration_,
        static_cast<uint16_t>((m_.hotel_edge_.checkout_time_ + offset) -
                              (start_time % MINUTES_A_DAY)));
    return edge_cost(duration, false, m_.hotel_edge_.price_);
  }
};

/* convenience helper functions to generate the right edge type */

inline edge make_route_edge(node* from, node* to,
                            mcd::vector<light_connection> const& connections) {
  return edge(from, to, connections);
}

inline edge make_foot_edge(node* from, node* to, uint16_t time_cost = 0,
                           bool transfer = false) {
  return edge(from, to, edge::FOOT_EDGE, time_cost, 0, transfer);
}

inline edge make_after_train_fwd_edge(node* from, node* to,
                                      uint16_t time_cost = 0,
                                      bool transfer = false) {
  return edge(from, to, edge::AFTER_TRAIN_FWD_EDGE, time_cost, 0, transfer);
}

inline edge make_after_train_bwd_edge(node* from, node* to,
                                      uint16_t time_cost = 0,
                                      bool transfer = false) {
  return edge(from, to, edge::AFTER_TRAIN_BWD_EDGE, time_cost, 0, transfer);
}

inline edge make_mumo_edge(node* from, node* to, uint16_t time_cost = 0,
                           uint16_t price = 0, uint16_t accessibility = 0,
                           int mumo_id = 0) {
  return edge(from, to, edge::MUMO_EDGE, time_cost, price, false, mumo_id, 0, 0,
              accessibility);
}

inline edge make_time_dependent_mumo_edge(node* from, node* to,
                                          uint16_t time_cost, uint16_t price,
                                          int mumo_id, uint16_t interval_begin,
                                          uint16_t interval_end) {
  return edge(from, to, edge::TIME_DEPENDENT_MUMO_EDGE, time_cost, price, false,
              mumo_id, interval_begin, interval_end);
}

inline edge make_periodic_mumo_edge(node* from, node* to, uint16_t time_cost,
                                    uint16_t price, int mumo_id,
                                    uint16_t interval_begin,
                                    uint16_t interval_end) {
  return edge(from, to, edge::PERIODIC_MUMO_EDGE, time_cost, price, false,
              mumo_id, interval_begin, interval_end);
}

inline edge make_hotel_edge(node* station_node, uint16_t checkout_time,
                            uint16_t min_stay_duration, uint16_t price,
                            int mumo_id) {
  return edge(station_node, checkout_time, min_stay_duration, price, mumo_id);
}

inline edge make_invalid_edge(node* from, node* to) {
  return edge(from, to, edge::INVALID_EDGE, 0, 0, false, 0);
}

inline edge make_through_edge(node* from, node* to) {
  return edge(from, to, edge::THROUGH_EDGE, 0, 0, false, 0);
}

inline edge make_enter_edge(node* from, node* to, uint16_t time_cost = 0,
                            bool transfer = false) {
  return edge(from, to, edge::ENTER_EDGE, time_cost, 0, transfer);
}

inline edge make_exit_edge(node* from, node* to, uint16_t time_cost = 0,
                           bool transfer = false) {
  return edge(from, to, edge::EXIT_EDGE, time_cost, 0, transfer);
}

inline edge make_fwd_edge(node* from, node* to, uint16_t time_cost = 0,
                          bool transfer = false) {
  return edge(from, to, edge::FWD_EDGE, time_cost, 0, transfer);
}

inline edge make_bwd_edge(node* from, node* to, uint16_t time_cost = 0,
                          bool transfer = false) {
  return edge(from, to, edge::BWD_EDGE, time_cost, 0, transfer);
}

}  // namespace motis
