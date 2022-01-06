#pragma once

#include <cassert>
#include <algorithm>

#include "motis/variant.h"
#include "motis/vector.h"

#include "utl/overloaded.h"

#include "motis/core/common/constants.h"
#include "motis/core/schedule/bitfield.h"
#include "motis/core/schedule/connection.h"
#include "motis/core/schedule/time.h"

namespace motis {

struct node;

struct edge_cost {
  edge_cost() = default;

  constexpr edge_cost(duration_t const time, static_light_connection const* c,
                      day_idx_t const day)
      : static_lcon_(c), day_(day), time_(time) {}

  constexpr edge_cost(duration_t const time, rt_light_connection const* c)
      : rt_lcon_(c), time_(time) {}

  constexpr explicit edge_cost(duration_t time, bool transfer = false,
                               uint16_t price = 0, uint16_t accessibility = 0)
      : static_lcon_(nullptr),
        rt_lcon_(nullptr),
        day_(-1),
        time_(time),
        price_(price),
        transfer_(transfer),
        accessibility_(accessibility) {}

  constexpr bool is_valid() const { return time_ != INVALID_DURATION; }

  static_light_connection const* static_lcon_{nullptr};
  rt_light_connection const* rt_lcon_{nullptr};
  day_idx_t day_{-1};
  duration_t time_{INVALID_DURATION};
  uint16_t price_{0U};
  bool transfer_{false};
  uint16_t accessibility_{0U};
};

constexpr auto const NO_EDGE = edge_cost();
constexpr auto const FREE_EDGE = edge_cost(0, false, 0, 0);

enum class search_dir { FWD, BWD };

enum class edge_type : uint8_t {
  INVALID_EDGE,
  STATIC_ROUTE_EDGE,
  RT_ROUTE_EDGE,
  FOOT_EDGE,
  AFTER_TRAIN_FWD_EDGE,
  AFTER_TRAIN_BWD_EDGE,
  MUMO_EDGE,
  THROUGH_EDGE,
  ENTER_EDGE,
  EXIT_EDGE,
  FWD_EDGE,
  BWD_EDGE,
  NUM_EDGE_TYPES
};

struct edge {
  struct invalid_edge {
    ptr<node> from_{nullptr};
    ptr<node> to_{nullptr};
  };

  struct static_route_edge {
    explicit static_route_edge(ptr<node> from, ptr<node> to,
                               mcd::vector<static_light_connection>&& conns)
        : from_{from}, to_{to}, conns_{std::move(conns)} {
      std::sort(std::begin(conns_), std::end(conns_));
    }

    template <search_dir Dir = search_dir::FWD>
    edge_cost get_edge_cost(time const start_time) const {
      auto const [c, day] = get_connection<Dir>(start_time);
      return (c == nullptr)
                 ? NO_EDGE
                 : edge_cost(
                       (Dir == search_dir::FWD)
                           ? c->event_time(event_type::ARR, day) - start_time
                           : start_time - c->event_time(event_type::DEP, day),
                       c, day);
    }

    template <search_dir Dir = search_dir::FWD>
    std::pair<static_light_connection const*, day_idx_t> get_connection(
        time const start_time) const {
      if (conns_.empty()) {
        return {nullptr, 0};
      }

      if constexpr (Dir == search_dir::FWD) {
        auto it = std::lower_bound(
            begin(conns_), std::end(conns_),
            static_light_connection{start_time.mam(), 0U}, d_time_lt{});

        auto const abort_time = start_time + MAX_TRAVEL_TIME_MINUTES;
        auto day = start_time.day();

        while (true) {
          if (day >= MAX_DAYS) {
            return {nullptr, 0};
          }

          if (it == end(conns_)) {
            it = begin(conns_);
            day += 1;
            continue;
          }

          if (it->event_time(event_type::DEP, day) > abort_time) {
            return {nullptr, 0};
          }

          if (it->traffic_days_->test(day)) {
            return {&*it, day};
          } else {
            ++it;
          }
        }
      } else {
        auto it = std::lower_bound(
            std::rbegin(conns_), std::rend(conns_),
            static_light_connection{0U, start_time.mam()}, a_time_gt{});

        auto const abort_time = start_time - MAX_TRAVEL_TIME_MINUTES;
        auto day = start_time.day();

        while (true) {
          if (day < 0) {
            return {nullptr, 0};
          }

          if (it == std::rend(conns_)) {
            it = std::rbegin(conns_);
            day -= 1;
            continue;
          }

          if (it->event_time(event_type::ARR, day) < abort_time) {
            return {nullptr, 0};
          }

          if (it->traffic_days_->test(day)) {
            return {&*it, day};
          } else {
            ++it;
          }
        }
      }
    }

    edge_cost get_min_edge_cost() const {
      if (conns_.empty()) {
        return NO_EDGE;
      } else {
        return edge_cost(std::min_element(begin(conns_), std::end(conns_),
                                          [](auto const& c1, auto const& c2) {
                                            return c1.travel_time() <
                                                   c2.travel_time();
                                          })
                             ->travel_time(),
                         false, begin(conns_)->full_con_->price_);
      }
    }

    ptr<node> from_{nullptr};
    ptr<node> to_{nullptr};
    mcd::vector<static_light_connection> conns_;
  };

  struct rt_route_edge {
    rt_route_edge(ptr<node> from, ptr<node> to,
                  mcd::vector<rt_light_connection>&& conns)
        : from_{from}, to_{to}, conns_{std::move(conns)} {
      std::sort(std::begin(conns_), std::end(conns_));
    }

    template <search_dir Dir = search_dir::FWD>
    edge_cost get_edge_cost(time const start_time) const {
      auto const* c = get_connection<Dir>(start_time);
      return (c == nullptr) ? NO_EDGE
                            : edge_cost((Dir == search_dir::FWD)
                                            ? time{c->a_time_} - start_time
                                            : start_time - time{c->d_time_},
                                        c);
    }

    template <search_dir Dir = search_dir::FWD>
    rt_light_connection const* get_connection(time const start_time) const {
      if (conns_.empty()) {
        return nullptr;
      }

      if (Dir == search_dir::FWD) {
        auto it = std::lower_bound(std::begin(conns_), std::end(conns_),
                                   rt_light_connection(start_time, time{}));

        if (it == std::end(conns_)) {
          return nullptr;
        } else {
          return get_next_valid_lcon(&*it);
        }
      } else {
        auto it = std::lower_bound(
            conns_.rbegin(), conns_.rend(),
            rt_light_connection(time{}, start_time),
            [](rt_light_connection const& lhs, rt_light_connection const& rhs) {
              return lhs.a_time_ > rhs.a_time_;
            });

        if (it == conns_.rend()) {
          return nullptr;
        } else {
          return get_prev_valid_lcon(&*it);
        }
      }
    }

    rt_light_connection const* get_next_valid_lcon(
        rt_light_connection const* lc, unsigned skip = 0) const {
      assert(lc != nullptr);

      auto it = lc;
      while (it != end(conns_)) {
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

    rt_light_connection const* get_prev_valid_lcon(
        rt_light_connection const* lc, unsigned skip = 0) const {
      assert(lc != nullptr);

      auto it = std::reverse_iterator<rt_light_connection const*>(lc);
      --it;
      while (it != conns_.rend()) {
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

    edge_cost get_min_edge_cost() const {
      if (conns_.empty()) {
        return NO_EDGE;
      } else {
        return edge_cost(std::min_element(begin(conns_), std::end(conns_),
                                          [](auto const& c1, auto const& c2) {
                                            return c1.travel_time() <
                                                   c2.travel_time();
                                          })
                             ->travel_time(),
                         false, begin(conns_)->full_con_->price_);
      }
    }

    ptr<node> from_{nullptr};
    ptr<node> to_{nullptr};
    mcd::vector<rt_light_connection> conns_;
  };

  struct constant_cost_edge {
    edge_cost get_edge_cost() const {
      return edge_cost(time_cost_, is_transfer_, price_, accessibility_);
    }

    edge_cost get_min_edge_cost() const { return edge_cost(0U, is_transfer_); }

    ptr<node> from_{nullptr};
    ptr<node> to_{nullptr};

    // edge weight
    uint16_t time_cost_{0};
    uint16_t price_{0};
    uint16_t accessibility_{0};
    bool is_transfer_{false};

    // id for mumo edge
    int32_t mumo_id_{-1};
  };

  using data_t = mcd::variant<invalid_edge,  // INVALID_EDGE
                              static_route_edge,  // STATIC_ROUTE_EDGE
                              rt_route_edge,  // RT_ROUTE_EDGE
                              constant_cost_edge,  // FOOT_EDGE
                              constant_cost_edge,  // AFTER_TRAIN_FWD_EDGE
                              constant_cost_edge,  // AFTER_TRAIN_BWD_EDGE
                              constant_cost_edge,  // MUMO_EDGE
                              constant_cost_edge,  // THROUGH_EDGE
                              constant_cost_edge,  // ENTER_EDGE
                              constant_cost_edge,  // EXIT_EDGE
                              constant_cost_edge,  // FWD_EDGE
                              constant_cost_edge  // BWD_EDGE
                              >;

  static_assert(static_cast<std::underlying_type_t<edge_type>>(
                    edge_type::NUM_EDGE_TYPES) == mcd::variant_size<data_t>());

  template <edge_type>
  struct id {};

  /**  invalid edge constructor */
  edge(node* from, node* to) { data_.emplace<invalid_edge>(from, to); }

  /** route edge constructor. */
  edge(node* from, node* to, mcd::vector<static_light_connection>&& conns) {
    data_.emplace<static_route_edge>(from, to, std::move(conns));
  }

  edge(node* from, node* to, mcd::vector<rt_light_connection>&& conns) {
    data_.emplace<rt_route_edge>(from, to, std::move(conns));
  }

  /** foot edge constructor. */
  template <edge_type EdgeType>
  edge(id<EdgeType>, node* from, node* to, uint16_t time_cost, uint16_t price,
       bool transfer, int mumo_id = 0, uint16_t accessibility = 0) {
    data_.emplace<to_index(EdgeType)>(constant_cost_edge{
        from, to, time_cost, price, accessibility, transfer, mumo_id});
  }

  constexpr static data_t::index_t to_index(edge_type const t) {
    return static_cast<data_t::index_t>(
        static_cast<std::underlying_type_t<edge_type>>(t));
  }

  template <search_dir Dir = search_dir::FWD>
  edge_cost get_edge_cost(time const start_time, bool const last_con) const {
    switch (type()) {
      case edge_type::STATIC_ROUTE_EDGE:
        return mcd::get<static_route_edge>(data_).get_edge_cost<Dir>(
            start_time);

      case edge_type::RT_ROUTE_EDGE:
        return mcd::get<rt_route_edge>(data_).get_edge_cost<Dir>(start_time);

      case edge_type::ENTER_EDGE:
        if (Dir == search_dir::FWD) {
          return FREE_EDGE;
        } else {
          return last_con ? mcd::get<constant_cost_edge>(data_).get_edge_cost()
                          : NO_EDGE;
        }

      case edge_type::EXIT_EDGE:
        if (Dir == search_dir::FWD) {
          return last_con ? mcd::get<constant_cost_edge>(data_).get_edge_cost()
                          : NO_EDGE;
        } else {
          return FREE_EDGE;
        }

      case edge_type::AFTER_TRAIN_FWD_EDGE:
        if (Dir == search_dir::FWD) {
          return last_con ? mcd::get<constant_cost_edge>(data_).get_edge_cost()
                          : NO_EDGE;
        } else {
          return NO_EDGE;
        }

      case edge_type::AFTER_TRAIN_BWD_EDGE:
        if (Dir == search_dir::BWD) {
          return last_con ? mcd::get<constant_cost_edge>(data_).get_edge_cost()
                          : NO_EDGE;
        } else {
          return NO_EDGE;
        }

      case edge_type::FWD_EDGE:
        if (Dir == search_dir::FWD) {
          return mcd::get<constant_cost_edge>(data_).get_edge_cost();
        } else {
          return NO_EDGE;
        }

      case edge_type::BWD_EDGE:
        if (Dir == search_dir::BWD) {
          return mcd::get<constant_cost_edge>(data_).get_edge_cost();
        } else {
          return NO_EDGE;
        }

      case edge_type::MUMO_EDGE:
      case edge_type::FOOT_EDGE:
        [[fallthrough]];
        return mcd::get<constant_cost_edge>(data_).get_edge_cost();

      case edge_type::THROUGH_EDGE: return last_con ? FREE_EDGE : NO_EDGE;

      default: return NO_EDGE;
    }
  }

  edge_cost get_minimum_cost() const {
    switch (type()) {
      case edge_type::INVALID_EDGE: return NO_EDGE;

      case edge_type::RT_ROUTE_EDGE:
        return mcd::get<rt_route_edge>(data_).get_min_edge_cost();

      case edge_type::STATIC_ROUTE_EDGE:
        return mcd::get<static_route_edge>(data_).get_min_edge_cost();

      case edge_type::FOOT_EDGE:
      case edge_type::AFTER_TRAIN_FWD_EDGE:
      case edge_type::AFTER_TRAIN_BWD_EDGE:
      case edge_type::ENTER_EDGE:
      case edge_type::EXIT_EDGE:
      case edge_type::BWD_EDGE:
        return mcd::get<constant_cost_edge>(data_).get_edge_cost();

      default: return FREE_EDGE;
    }
  }

  node const* from() const {
    return data_.apply([](auto&& x) { return x.from_; });
  }

  node const* to() const {
    return data_.apply([](auto&& x) { return x.to_; });
  }

  template <search_dir Dir = search_dir::FWD>
  node const* get_destination() const {
    return (Dir == search_dir::FWD) ? to() : from();
  }

  template <search_dir Dir = search_dir::FWD>
  node const* get_source() const {
    return (Dir == search_dir::FWD) ? from() : to();
  }

  node const* get_destination(search_dir dir = search_dir::FWD) const {
    return (dir == search_dir::FWD) ? to() : from();
  }

  node const* get_source(search_dir dir = search_dir::FWD) const {
    return (dir == search_dir::FWD) ? from() : to();
  }

  bool valid() const { return type() != edge_type::INVALID_EDGE; }

  edge_type type() const { return static_cast<edge_type>(data_.index()); }

  bool is_route_edge() const { return is_route_edge(type()); }

  static bool is_route_edge(edge_type const t) {
    return t == edge_type::STATIC_ROUTE_EDGE || t == edge_type::RT_ROUTE_EDGE;
  }

  char const* type_str() const {
    switch (type()) {
      case edge_type::INVALID_EDGE: return "INVALID";
      case edge_type::STATIC_ROUTE_EDGE: return "ROUTE_EDGE";
      case edge_type::RT_ROUTE_EDGE: return "RT_ROUTE_EDGE";
      case edge_type::FOOT_EDGE: return "FOOT_EDGE";
      case edge_type::AFTER_TRAIN_FWD_EDGE: return "AFTER_TRAIN_FWD_EDGE";
      case edge_type::AFTER_TRAIN_BWD_EDGE: return "AFTER_TRAIN_BWD_EDGE";
      case edge_type::MUMO_EDGE: return "MUMO_EDGE";
      case edge_type::THROUGH_EDGE: return "THROUGH_EDGE";
      case edge_type::ENTER_EDGE: return "ENTER_EDGE";
      case edge_type::EXIT_EDGE: return "EXIT_EDGE";
      case edge_type::FWD_EDGE: return "FWD_EDGE";
      case edge_type::BWD_EDGE: return "BWD_EDGE";
      case edge_type::NUM_EDGE_TYPES: return "???";
    }
  }

  int get_mumo_id() const {
    return type() == edge_type::MUMO_EDGE
               ? mcd::get<constant_cost_edge>(data_).mumo_id_
               : -1;
  }

  bool empty() const {
    switch (type()) {
      case edge_type::STATIC_ROUTE_EDGE:
        return mcd::get<static_route_edge>(data_).conns_.empty();
      case edge_type::RT_ROUTE_EDGE:
        return mcd::get<rt_route_edge>(data_).conns_.empty();
      default: return true;
    }
  }

  data_t data_{invalid_edge{nullptr, nullptr}};
};

/* convenience helper functions to generate the right edge type */

inline edge make_static_route_edge(
    node* from, node* to, mcd::vector<static_light_connection>&& connections) {
  return edge{from, to, std::move(connections)};
}

inline edge make_rt_route_edge(node* from, node* to,
                               mcd::vector<rt_light_connection>&& connections) {
  return edge{from, to, std::move(connections)};
}

inline edge make_foot_edge(node* from, node* to, uint16_t time_cost = 0,
                           bool transfer = false) {
  return edge{
      edge::id<edge_type::FOOT_EDGE>{}, from, to, time_cost, 0, transfer};
}

inline edge make_after_train_fwd_edge(node* from, node* to,
                                      uint16_t time_cost = 0,
                                      bool transfer = false) {
  return edge(edge::id<edge_type::AFTER_TRAIN_FWD_EDGE>{}, from, to, time_cost,
              0, transfer);
}

inline edge make_after_train_bwd_edge(node* from, node* to,
                                      uint16_t time_cost = 0,
                                      bool transfer = false) {
  return edge(edge::id<edge_type::AFTER_TRAIN_BWD_EDGE>{}, from, to, time_cost,
              0, transfer);
}

inline edge make_mumo_edge(node* from, node* to, uint16_t time_cost = 0,
                           uint16_t price = 0, uint16_t accessibility = 0,
                           int mumo_id = 0) {
  return edge(edge::id<edge_type::MUMO_EDGE>{}, from, to, time_cost, price,
              false, mumo_id, accessibility);
}

inline edge make_invalid_edge(node* from, node* to) { return edge(from, to); }

inline edge make_through_edge(node* from, node* to) {
  return edge(edge::id<edge_type::THROUGH_EDGE>{}, from, to, 0, 0, false, 0);
}

inline edge make_enter_edge(node* from, node* to, uint16_t time_cost = 0,
                            bool transfer = false) {
  return edge(edge::id<edge_type::ENTER_EDGE>{}, from, to, time_cost, 0,
              transfer);
}

inline edge make_exit_edge(node* from, node* to, uint16_t time_cost = 0,
                           bool transfer = false) {
  return edge(edge::id<edge_type::EXIT_EDGE>{}, from, to, time_cost, 0,
              transfer);
}

inline edge make_fwd_edge(node* from, node* to, uint16_t time_cost = 0,
                          bool transfer = false) {
  return edge(edge::id<edge_type::FWD_EDGE>{}, from, to, time_cost, 0,
              transfer);
}

inline edge make_bwd_edge(node* from, node* to, uint16_t time_cost = 0,
                          bool transfer = false) {
  return edge(edge::id<edge_type::BWD_EDGE>{}, from, to, time_cost, 0,
              transfer);
}

}  // namespace motis
