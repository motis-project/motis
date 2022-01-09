#pragma once

#include <utl/verify.h>
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
  constexpr edge_cost() = default;

  edge_cost(duration_t const time, generic_light_connection const& c)
      : connection_{c}, time_{time} {}

  constexpr ~edge_cost() = default;

  constexpr explicit edge_cost(duration_t time, bool transfer, uint16_t price,
                               uint16_t accessibility)
      : time_(time),
        price_(price),
        transfer_(transfer),
        accessibility_(accessibility) {}

  constexpr bool is_valid() const { return time_ != INVALID_DURATION; }

  generic_light_connection connection_;
  duration_t time_{INVALID_DURATION};
  uint16_t price_{0U};
  bool transfer_{false};
  uint16_t accessibility_{0U};
};

constexpr auto const NO_EDGE = edge_cost{};
constexpr auto const FREE_EDGE = edge_cost{0U, false, 0U, 0U};

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
    template <search_dir Dir = search_dir::FWD>
    edge_cost get_edge_cost(time const start_time) const {
      auto const [c, day] = get_connection<Dir>(start_time);
      return (c == nullptr)
                 ? NO_EDGE
                 : edge_cost(
                       (Dir == search_dir::FWD)
                           ? c->event_time(event_type::ARR, day) - start_time
                           : start_time - c->event_time(event_type::DEP, day),
                       c, c->full_con_->price_, 0U);
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
                         false, begin(conns_)->full_con_->price_, 0U);
      }
    }

    ptr<node> from_{nullptr};
    ptr<node> to_{nullptr};
    mcd::vector<static_light_connection> conns_;
  };

  struct rt_route_edge {
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
                                   rt_light_connection(start_time, time{}),
                                   d_time_lt{});

        if (it == std::end(conns_)) {
          return nullptr;
        } else {
          return get_next_valid_lcon(&*it);
        }
      } else {
        auto it = std::lower_bound(conns_.rbegin(), conns_.rend(),
                                   rt_light_connection(time{}, start_time),
                                   a_time_gt{});

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
                         false, begin(conns_)->full_con_->price_, 0U);
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

    edge_cost get_min_edge_cost() const {
      return edge_cost(0U, is_transfer_, 0U, 0U);
    }

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

  node* from() const {
    return data_.apply([](auto&& x) { return x.from_; });
  }

  node* to() const {
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
    return "UNREACHABLE";
  }

  int get_mumo_id() const {
    return type() == edge_type::MUMO_EDGE
               ? mcd::get<constant_cost_edge>(data_).mumo_id_
               : -1;
  }

  duration_t constant_time_cost() const {
    utl::verify(mcd::holds_alternative<constant_cost_edge>(data_),
                "no constant time cost for {}", type_str());
    return mcd::get<constant_cost_edge>(data_).time_cost_;
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

  mcd::vector<static_light_connection>& static_lcons() {
    utl::verify(type() == edge_type::STATIC_ROUTE_EDGE,
                "static_lcons() called on {}", type_str());
    return mcd::get<static_route_edge>(data_).conns_;
  }

  mcd::vector<rt_light_connection>& rt_lcons() {
    utl::verify(type() == edge_type::RT_ROUTE_EDGE,
                "static_lcons() called on {}", type_str());
    return mcd::get<rt_route_edge>(data_).conns_;
  }

  mcd::vector<static_light_connection> const& static_lcons() const {
    utl::verify(type() == edge_type::STATIC_ROUTE_EDGE,
                "static_lcons() called on {}", type_str());
    return mcd::get<static_route_edge>(data_).conns_;
  }

  mcd::vector<rt_light_connection> const& rt_lcons() const {
    utl::verify(type() == edge_type::RT_ROUTE_EDGE,
                "static_lcons() called on {}", type_str());
    return mcd::get<rt_route_edge>(data_).conns_;
  }

  lcon_idx_t get_lcon_index(generic_light_connection const& glcon) const {
    switch (type()) {
      case edge_type::RT_ROUTE_EDGE: {
        auto const c = glcon.rt_con();
        auto const& conns = mcd::get<rt_route_edge>(data_).conns_;
        utl::verify(c >= begin(conns) && c < end(conns),
                    "get_lcon_index(): not found");
        return static_cast<lcon_idx_t>(std::distance(begin(conns), c));
      }

      case edge_type::STATIC_ROUTE_EDGE: {
        auto const c = glcon.static_con().first;
        auto const& conns = mcd::get<static_route_edge>(data_).conns_;
        utl::verify(c >= begin(conns) && c < end(conns),
                    "get_lcon_index(): not found");
        return static_cast<lcon_idx_t>(std::distance(begin(conns), c));
      }

      default: throw utl::fail("get_lcon_index on {}", type_str());
    }
  }

  bool contains(generic_light_connection const& c) const {
    switch (type()) {
      case edge_type::RT_ROUTE_EDGE:
        return c.is_rt() &&
               mcd::get<rt_route_edge>(data_).conns_.contains(c.rt_con());

      case edge_type::STATIC_ROUTE_EDGE:
        return c.is_static() &&
               mcd::get<static_route_edge>(data_).conns_.contains(
                   c.static_con().first);

      default: return false;
    }
  }

  template <search_dir Dir = search_dir::FWD>
  generic_light_connection get_connection(time const t) const {
    switch (type()) {
      case edge_type::RT_ROUTE_EDGE:
        return generic_light_connection{
            mcd::get<rt_route_edge>(data_).get_connection<Dir>(t)};

      case edge_type::STATIC_ROUTE_EDGE:
        return generic_light_connection{
            mcd::get<static_route_edge>(data_).get_connection<Dir>(t)};

      default: throw utl::fail("get_connection on {}", type_str());
    }
  }

  bool is_sorted() const {
    auto const is_sorted_dep = [](auto&& lcons) {
      return std::is_sorted(begin(lcons), end(lcons), [](auto&& a, auto&& b) {
        return a.d_time_ < b.d_time_;
      });
    };
    auto const is_sorted_arr = [](auto&& lcons) {
      return std::is_sorted(begin(lcons), end(lcons), [](auto&& a, auto&& b) {
        return a.a_time_ < b.a_time_;
      });
    };

    switch (type()) {
      case edge_type::RT_ROUTE_EDGE:
        return is_sorted_dep(mcd::get<rt_route_edge>(data_).conns_) &&
               is_sorted_arr(mcd::get<rt_route_edge>(data_).conns_);
      case edge_type::STATIC_ROUTE_EDGE:
        return is_sorted_dep(mcd::get<static_route_edge>(data_).conns_) &&
               is_sorted_arr(mcd::get<static_route_edge>(data_).conns_);
      default: return true;
    }
  }

  data_t data_{invalid_edge{nullptr, nullptr}};
};

/* convenience helper functions to generate the right edge type */

/** foot edge constructor. */
inline edge make_static_route_edge(
    node* from, node* to, mcd::vector<static_light_connection>&& conns) {
  edge e;
  std::sort(std::begin(conns), std::end(conns), d_time_lt{});
  e.data_.emplace<edge::static_route_edge>(edge::static_route_edge{
      .from_ = from, .to_ = to, .conns_ = std::move(conns)});
  return e;
}

inline edge make_rt_route_edge(node* from, node* to,
                               mcd::vector<rt_light_connection>&& conns) {
  edge e;
  std::sort(std::begin(conns), std::end(conns), d_time_lt{});
  e.data_.emplace<edge::rt_route_edge>(edge::rt_route_edge{
      .from_ = from, .to_ = to, .conns_ = std::move(conns)});
  return e;
}

inline edge make_foot_edge(node* from, node* to, uint16_t time_cost = 0,
                           bool transfer = false) {
  edge e;
  e.data_.emplace<edge::to_index(edge_type::FOOT_EDGE)>(
      edge::constant_cost_edge{.from_ = from,
                               .to_ = to,
                               .time_cost_ = time_cost,
                               .price_ = 0,
                               .accessibility_ = 0,
                               .is_transfer_ = transfer});
  return e;
}

inline edge make_after_train_fwd_edge(node* from, node* to,
                                      uint16_t time_cost = 0,
                                      bool transfer = false) {

  edge e;
  e.data_.emplace<edge::to_index(edge_type::AFTER_TRAIN_FWD_EDGE)>(
      edge::constant_cost_edge{.from_ = from,
                               .to_ = to,
                               .time_cost_ = time_cost,
                               .price_ = 0,
                               .accessibility_ = 0,
                               .is_transfer_ = transfer});
  return e;
}

inline edge make_after_train_bwd_edge(node* from, node* to,
                                      uint16_t time_cost = 0,
                                      bool transfer = false) {
  edge e;
  e.data_.emplace<edge::to_index(edge_type::AFTER_TRAIN_BWD_EDGE)>(
      edge::constant_cost_edge{.from_ = from,
                               .to_ = to,
                               .time_cost_ = time_cost,
                               .price_ = 0,
                               .accessibility_ = 0,
                               .is_transfer_ = transfer});
  return e;
}

inline edge make_mumo_edge(node* from, node* to, uint16_t time_cost = 0,
                           uint16_t price = 0, uint16_t accessibility = 0,
                           int mumo_id = 0) {
  edge e;
  e.data_.emplace<edge::to_index(edge_type::MUMO_EDGE)>(
      edge::constant_cost_edge{.from_ = from,
                               .to_ = to,
                               .time_cost_ = time_cost,
                               .price_ = price,
                               .accessibility_ = accessibility,
                               .is_transfer_ = false,
                               .mumo_id_ = mumo_id});
  return e;
}

inline edge make_invalid_edge(node* from, node* to) {
  edge e;
  e.data_.emplace<edge::invalid_edge>(from, to);
  return e;
}

inline edge make_through_edge(node* from, node* to) {
  edge e;
  e.data_.emplace<edge::to_index(edge_type::THROUGH_EDGE)>(
      edge::constant_cost_edge{.from_ = from,
                               .to_ = to,
                               .time_cost_ = 0,
                               .price_ = 0,
                               .accessibility_ = 0,
                               .is_transfer_ = false,
                               .mumo_id_ = 0});
  return e;
}

inline edge make_enter_edge(node* from, node* to, uint16_t time_cost = 0,
                            bool transfer = false) {
  edge e;
  e.data_.emplace<edge::to_index(edge_type::ENTER_EDGE)>(
      edge::constant_cost_edge{.from_ = from,
                               .to_ = to,
                               .time_cost_ = time_cost,
                               .price_ = 0,
                               .accessibility_ = 0,
                               .is_transfer_ = transfer});
  return e;
}

inline edge make_exit_edge(node* from, node* to, uint16_t time_cost = 0,
                           bool transfer = false) {
  edge e;
  e.data_.emplace<edge::to_index(edge_type::EXIT_EDGE)>(
      edge::constant_cost_edge{.from_ = from,
                               .to_ = to,
                               .time_cost_ = time_cost,
                               .price_ = 0,
                               .accessibility_ = 0,
                               .is_transfer_ = transfer});
  return e;
}

inline edge make_fwd_edge(node* from, node* to, uint16_t time_cost = 0,
                          bool transfer = false) {
  edge e;
  e.data_.emplace<edge::to_index(edge_type::FWD_EDGE)>(
      edge::constant_cost_edge{.from_ = from,
                               .to_ = to,
                               .time_cost_ = time_cost,
                               .price_ = 0,
                               .accessibility_ = 0,
                               .is_transfer_ = transfer});
  return e;
}

inline edge make_bwd_edge(node* from, node* to, uint16_t time_cost = 0,
                          bool transfer = false) {
  edge e;
  e.data_.emplace<edge::to_index(edge_type::BWD_EDGE)>(
      edge::constant_cost_edge{.from_ = from,
                               .to_ = to,
                               .time_cost_ = time_cost,
                               .price_ = 0,
                               .accessibility_ = 0,
                               .is_transfer_ = transfer});
  return e;
}

}  // namespace motis
