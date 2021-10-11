#pragma once

#include <cassert>
#include <iostream>
#include <type_traits>

#include "cista/reflection/comparable.h"

#include "motis/core/common/hash_helper.h"
#include "motis/core/schedule/edges.h"
#include "motis/core/schedule/event_type.h"
#include "motis/core/schedule/time.h"
#include "motis/core/schedule/trip.h"

namespace motis {

struct node;

struct ev_key {
  CISTA_COMPARABLE()

  bool is_not_null() const { return route_edge_.is_not_null(); }

  bool is_arrival() const { return ev_type_ == event_type::ARR; }
  bool is_departure() const { return ev_type_ == event_type::DEP; }

  bool lcon_is_valid() const { return is_not_null() && lcon()->valid_ != 0U; }

  explicit operator bool() const { return is_not_null(); }

  ev_key get_opposite() const {
    auto const ev_type =
        ev_type_ == event_type::ARR ? event_type::DEP : event_type::ARR;
    return {route_edge_, lcon_idx_, ev_type, day_};
  }

  light_connection const* lcon() const {
    return &route_edge_->m_.route_edge_.conns_[lcon_idx_];
  }

  time get_time() const { return lcon()->event_time(ev_type_, day_); }

  bool is_canceled() const { return lcon()->valid_ == 0U; }

  int get_track() const {
    auto const full_con = lcon()->full_con_;
    return is_arrival() ? full_con->a_track_ : full_con->d_track_;
  }

  node* get_node() const {
    return ev_type_ == event_type::DEP ? route_edge_->from_ : route_edge_->to_;
  }

  uint32_t get_station_idx() const { return get_node()->get_station()->id_; }

  cista::hash_t hash() const {
    return cista::build_hash(route_edge_, lcon_idx_, ev_type_, day_);
  }

  trip::route_edge route_edge_{nullptr};
  lcon_idx_t lcon_idx_{0};
  event_type ev_type_{event_type::DEP};
  day_idx_t day_{0};
};

}  // namespace motis
