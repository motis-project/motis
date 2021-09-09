#pragma once

#include <algorithm>
#include <set>
#include <tuple>

#include "utl/to_set.h"

#include "motis/routing/label/criteria/transfers.h"
#include "motis/routing/label/criteria/travel_time.h"
#include "motis/routing/label/dominance.h"
#include "motis/routing/label/tie_breakers.h"

#include "motis/protocol/RoutingResponse_generated.h"

namespace motis::eval::comparator {

enum connection_accessors { DURATION, TRANSFERS, PRICE };

struct journey_meta_data {
  explicit journey_meta_data(Connection const* c)
      : c_(c),
        departure_time_(c->stops()->size() != 0
                            ? c->stops()->begin()->departure()->time()
                            : 0),
        arrival_time_(
            c->stops()->size() != 0
                ? c->stops()->Get(c->stops()->size() - 1)->arrival()->time()
                : 0),
        duration_(arrival_time_ - departure_time_),
        transfers_(
            std::max(0, static_cast<int>(std::count_if(
                            std::begin(*c->stops()), std::end(*c->stops()),
                            [](Stop const* s) { return s->exit(); })) -
                            1)) {}

  inline friend bool operator==(journey_meta_data const& a,
                                journey_meta_data const& b) {
    return a.as_tuple() == b.as_tuple();
  }

  inline friend bool operator<(journey_meta_data const& a,
                               journey_meta_data const& b) {
    return a.as_tuple() < b.as_tuple();
  }

  inline time_t get_departure_time() const { return departure_time_; }

  inline time_t get_arrival_time() const { return arrival_time_; }

  inline std::tuple<int, int, int> as_tuple() const {
    return std::make_tuple(duration_, get_departure_time(), transfers_);
  }

  inline bool dominates(journey_meta_data const& o) const {
    return departure_time_ >= o.departure_time_ &&
           arrival_time_ <= o.arrival_time_ &&  //
           transfers_ <= o.transfers_;
  }

  bool valid() const { return c_->stops()->size() != 0; }

  Connection const* c_;
  time_t departure_time_;
  time_t arrival_time_;
  unsigned duration_;
  unsigned transfers_;
};

struct response {
  response(std::vector<Connection const*> const& c,
           routing::RoutingResponse const* r)
      : connections_{utl::to_set(
            c, [](auto&& con) { return journey_meta_data(con); })},
        r_{r} {}

  explicit response(routing::RoutingResponse const* r)
      : connections_{utl::to_set(
            *r->connections(),
            [](Connection const* c) { return journey_meta_data(c); })},
        r_{r} {}

  bool valid() const {
    return std::all_of(begin(connections_), end(connections_),
                       [](journey_meta_data const& c) { return c.valid(); });
  }

  std::set<journey_meta_data> connections_;
  routing::RoutingResponse const* r_;
};

}  // namespace motis::eval::comparator
