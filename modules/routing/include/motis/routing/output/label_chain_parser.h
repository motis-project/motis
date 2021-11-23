#pragma once

#include <utility>
#include <vector>

#include "utl/verify.h"

#include "motis/core/schedule/connection.h"
#include "motis/core/access/realtime_access.h"

#include "motis/routing/output/stop.h"
#include "motis/routing/output/transport.h"

constexpr auto MOTIS_UNKNOWN_TRACK = 0;

namespace motis::routing::output {

enum state {
  AT_STATION,
  PRE_CONNECTION,
  IN_CONNECTION,
  PRE_WALK,
  WALK,
  WALK_SKIP,
  BWD_WALK,
  BWD_WALK_SKIP,
  IN_CONNECTION_THROUGH,
  IN_CONNECTION_THROUGH_SKIP,
  ONTRIP_TRAIN_START
};

template <typename Label>
inline node* get_node(Label const& l) {
  return l.edge_->to_;
}

template <typename Label>
state next_state(int s, Label const* c, Label const* n) {
  switch (s) {
    case AT_STATION:
      if (n && get_node(*n)->is_station_node()) {
        return WALK;
      } else {
        return PRE_CONNECTION;
      }
    case PRE_CONNECTION:
    case IN_CONNECTION_THROUGH: return IN_CONNECTION;
    case IN_CONNECTION_THROUGH_SKIP:
      return get_node(*c)->is_foot_node() ? WALK : AT_STATION;
    case IN_CONNECTION:
      if (c->connection_ == nullptr) {
        if (n && get_node(*c)->type() == node_type::STATION_NODE &&
            get_node(*n)->type() == node_type::FOOT_NODE) {
          if (get_node(*c)->get_station() == get_node(*n)->get_station()) {
            return WALK_SKIP;
          } else {
            return BWD_WALK;
          }
        }

        switch (get_node(*c)->type()) {
          case node_type::STATION_NODE:
            return n && get_node(*n)->is_station_node() ? WALK : AT_STATION;
          case node_type::FOOT_NODE: return WALK;
          case node_type::ROUTE_NODE:
            return get_node(*n)->is_route_node() ? IN_CONNECTION_THROUGH
                                                 : IN_CONNECTION_THROUGH_SKIP;
          case node_type::PLATFORM_NODE: return AT_STATION;
        }
      } else {
        return IN_CONNECTION;
      }
      break;
    case WALK:
      if (n && get_node(*n)->is_foot_node()) {
        if (get_node(*c)->get_station() == get_node(*n)->get_station()) {
          return WALK_SKIP;
        } else {
          return BWD_WALK;
        }
      } else if (n && get_node(*n)->is_station_node()) {
        return WALK;
      } else {
        return AT_STATION;
      }
    case BWD_WALK:
      if (n && get_node(*n)->is_station_node()) {
        return BWD_WALK_SKIP;
      } else {
        return AT_STATION;
      }
    case BWD_WALK_SKIP:
      if ((n && get_node(*n)->type() == node_type::ROUTE_NODE) ||
          (!n && get_node(*c)->type() == node_type::STATION_NODE)) {
        return AT_STATION;
      }
    case WALK_SKIP:
      if (n && get_node(*c)->type() == node_type::STATION_NODE &&
          get_node(*n)->type() == node_type::FOOT_NODE &&
          get_node(*c)->get_station() != get_node(*n)->get_station()) {
        return BWD_WALK;
      } else {
        return WALK;
      }
    case ONTRIP_TRAIN_START: return IN_CONNECTION;
  }
  return static_cast<state>(s);
};

template <typename LabelIt>
int initial_state(LabelIt& it) {
  if (get_node(*it)->is_route_node()) {
    if (get_node(*std::next(it))->is_station_node() ||
        get_node(*std::next(it))->is_platform_node()) {
      ++it;
      return AT_STATION;
    } else if (get_node(*std::next(it))->is_foot_node()) {
      ++it;
      return WALK;
    } else {
      return ONTRIP_TRAIN_START;
    }
  } else if (get_node(*std::next(it))->is_station_node()) {
    return WALK;
  } else if (get_node(*std::next(it))->is_foot_node()) {
    if (get_node(*it)->get_station() ==
        get_node(*std::next(it))->get_station()) {
      ++it;
    }
    return WALK;
  } else {
    return AT_STATION;
  }
}

template <typename Label>
std::pair<std::vector<intermediate::stop>, std::vector<intermediate::transport>>
parse_label_chain(schedule const& sched, Label* terminal_label,
                  search_dir const dir) {
  std::vector<Label> labels;

  auto c = terminal_label;
  do {
    labels.insert(begin(labels), *c);
  } while ((c = c->pred_));

  auto last_node =
      make_node(node_type::ROUTE_NODE, sched.station_nodes_.at(0).get(), 0, 0);
  edge last_edge;
  if (dir == search_dir::BWD) {
    for (auto i = 0UL; i < labels.size() - 1; ++i) {
      labels[i].edge_ = labels[i + 1].edge_;
      labels[i].connection_ = labels[i + 1].connection_;
    }

    auto& last_label = labels[labels.size() - 1];
    auto const second_edge = last_label.edge_;
    last_edge = make_invalid_edge(&last_node, second_edge->from_);
    last_label.edge_ = &last_edge;
    last_label.connection_ = nullptr;

    std::reverse(begin(labels), end(labels));
  }

  std::pair<std::vector<intermediate::stop>,
            std::vector<intermediate::transport>>
      ret;
  auto& stops = ret.first;
  auto& transports = ret.second;

  node const* last_route_node = nullptr;
  light_connection const* last_con = nullptr;
  auto walk_arrival = INVALID_TIME;
  auto walk_arrival_di = delay_info{{nullptr, INVALID_TIME, event_type::DEP}};
  auto stop_index = -1;

  auto it = begin(labels);
  int current_state = initial_state(it);
  while (it != end(labels)) {
    auto& current = *it;

    switch (current_state) {
      case ONTRIP_TRAIN_START:
      case AT_STATION: {
        if (current.edge_->type() == edge::HOTEL_EDGE &&
            get_node(*std::next(it))->is_foot_node()) {
          break;
        }
        unsigned a_track = MOTIS_UNKNOWN_TRACK,
                 a_sched_track = MOTIS_UNKNOWN_TRACK;
        unsigned d_track = MOTIS_UNKNOWN_TRACK,
                 d_sched_track = MOTIS_UNKNOWN_TRACK;
        time a_time = walk_arrival, a_sched_time = walk_arrival;
        time d_time = INVALID_TIME, d_sched_time = INVALID_TIME;
        timestamp_reason a_reason = walk_arrival_di.get_reason(),
                         d_reason = timestamp_reason::SCHEDULE;
        if (a_time == INVALID_TIME && last_con != nullptr) {
          a_track = last_con->full_con_->a_track_;
          a_time = last_con->a_time_;

          auto a_di =
              get_delay_info(sched, last_route_node, last_con, event_type::ARR);
          a_sched_time = a_di.get_schedule_time();
          a_reason = a_di.get_reason();
          a_sched_track = get_schedule_track(sched, last_route_node, last_con,
                                             event_type::ARR);
        }

        walk_arrival = INVALID_TIME;

        auto inc = current_state == AT_STATION ? 1 : 0;
        auto s1 = std::next(it, 0 + inc);
        if (s1 != end(labels)) {
          auto s2 = std::next(it, 1 + inc);

          if (current_state == ONTRIP_TRAIN_START && s2 != end(labels) &&
              s2->edge_->type() == edge::THROUGH_EDGE) {
            s1 = std::next(it, 1 + inc);
            s2 = std::next(it, 2 + inc);
            ++it;  // skip the initial through edge entirely
          }

          if (s2 != end(labels) && s2->connection_ != nullptr) {
            auto const& succ = *s2;
            d_track = succ.connection_->full_con_->d_track_;
            d_time = succ.connection_->d_time_;

            auto d_di = get_delay_info(sched, get_node(*s1), succ.connection_,
                                       event_type::DEP);
            d_sched_time = d_di.get_schedule_time();
            d_reason = d_di.get_reason();
            d_sched_track = get_schedule_track(
                sched, get_node(*s1), succ.connection_, event_type::DEP);
          }
        }

        stops.emplace_back(static_cast<unsigned int>(++stop_index),
                           get_node(current)->get_station()->id_, a_track,
                           d_track, a_sched_track, d_sched_track, a_time,
                           d_time, a_sched_time, d_sched_time, a_reason,
                           d_reason,
                           last_con != nullptr && a_time != INVALID_TIME,
                           d_time != INVALID_TIME);
        break;
      }

      case BWD_WALK:
      case WALK: {
        utl::verify(std::next(it) != end(labels),
                    "label chain parser in state walk at last label");

        if (last_con != nullptr) {
          walk_arrival_di =
              get_delay_info(sched, last_route_node, last_con, event_type::ARR);
        }

        stops.emplace_back(static_cast<unsigned int>(++stop_index),
                           get_node(current)->get_station()->id_,
                           last_con == nullptr ? MOTIS_UNKNOWN_TRACK
                                               : last_con->full_con_->a_track_,
                           MOTIS_UNKNOWN_TRACK,

                           last_con == nullptr
                               ? MOTIS_UNKNOWN_TRACK
                               : get_schedule_track(sched, last_route_node,
                                                    last_con, event_type::ARR),
                           MOTIS_UNKNOWN_TRACK,

                           // Arrival graph time:
                           stops.empty() ? INVALID_TIME
                           : last_con    ? last_con->a_time_
                                         : current.now_,

                           // Departure graph time:
                           current.now_,

                           // Arrival schedule time:
                           stops.empty() ? INVALID_TIME
                           : last_con    ? walk_arrival_di.get_schedule_time()
                                         : current.now_,

                           // Departure schedule time:
                           current.now_,

                           // Arrival reason timestamp
                           walk_arrival_di.get_reason(),

                           // Departure reason timestamp
                           walk_arrival_di.get_reason(),

                           // Leaving
                           last_con != nullptr,

                           // Entering
                           false);

        transports.emplace_back(
            stop_index, static_cast<unsigned int>(stop_index) + 1,
            std::next(it)->now_ - current.now_,
            std::next(it)->edge_->get_mumo_id(), 0,
            std::next(it)->edge_->get_minimum_cost().accessibility_);

        walk_arrival = std::next(it)->now_;
        last_con = nullptr;
        break;
      }

      case IN_CONNECTION: {
        if (current.connection_) {
          transports.emplace_back(static_cast<unsigned int>(stop_index),
                                  static_cast<unsigned int>(stop_index) + 1,
                                  current.connection_);
        }

        // do not collect the last connection route node.
        assert(std::next(it) != end(labels));
        auto succ = *std::next(it);

        if (get_node(succ)->is_route_node()) {
          auto dep_route_node = get_node(current);

          // skip through edge.
          if (!succ.connection_) {
            dep_route_node = get_node(succ);
            succ = *std::next(it, 2);
          }

          // through edge used but not the route edge after that
          // (instead: went to station node using the leaving edge)
          if (succ.connection_) {
            auto const a_di = get_delay_info(
                sched, get_node(current), current.connection_, event_type::ARR);
            auto const d_di = get_delay_info(sched, dep_route_node,
                                             succ.connection_, event_type::DEP);
            auto const a_sched_track = get_schedule_track(
                sched, get_node(current), current.connection_, event_type::ARR);
            auto const d_sched_track = get_schedule_track(
                sched, dep_route_node, succ.connection_, event_type::DEP);

            stops.emplace_back(
                static_cast<unsigned int>(++stop_index),
                get_node(current)->get_station()->id_,
                current.connection_->full_con_->a_track_,  // NOLINT
                succ.connection_->full_con_->d_track_, a_sched_track,
                d_sched_track, current.connection_->a_time_,
                succ.connection_->d_time_, a_di.get_schedule_time(),
                d_di.get_schedule_time(), a_di.get_reason(), d_di.get_reason(),
                false, false);
          }
        }

        last_route_node = get_node(current);
        last_con = current.connection_;
        break;
      }
    }

    ++it;
    if (it != end(labels)) {
      current = *it;
      auto next = it == end(labels) || std::next(it) == end(labels)
                      ? nullptr
                      : &(*std::next(it));
      current_state = next_state(current_state, &current, next);
    }
  }

  if (dir == search_dir::BWD && transports.back().is_walk()) {
    utl::verify(stops.size() > 1,
                "Less than two intermediate stops in label chain parser");
    auto& second_to_last = stops[stops.size() - 2];
    auto& last = stops[stops.size() - 1];

    second_to_last.d_time_ = second_to_last.a_time_;
    second_to_last.d_sched_time_ = second_to_last.a_sched_time_;

    auto const walk_duration = transports.back().duration_;
    last.a_time_ = second_to_last.d_time_ + walk_duration;
    last.a_sched_time_ = second_to_last.d_sched_time_ + walk_duration;
  }

  return ret;
}

}  // namespace motis::routing::output
