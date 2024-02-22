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

constexpr auto const kTracing = false;

template <typename... T>
void trace(fmt::format_string<T...> str, T&&... t) {
  if (kTracing) {
    fmt::print(std::cerr, str, std::forward<T>(t)...);
  }
}

enum class state {
  AT_STATION,
  PRE_CONNECTION,
  IN_CONNECTION,
  WALK,
  WALK_SKIP,
  BWD_WALK,
  BWD_WALK_SKIP,
  IN_CONNECTION_THROUGH,
  IN_CONNECTION_THROUGH_SKIP,
  ONTRIP_TRAIN_START
};

char const* state_to_str(state const s) {
  constexpr char const* const strs[] = {"AT_STATION",
                                        "PRE_CONNECTION",
                                        "IN_CONNECTION",
                                        "WALK",
                                        "WALK_SKIP",
                                        "BWD_WALK",
                                        "BWD_WALK_SKIP",
                                        "IN_CONNECTION_THROUGH",
                                        "IN_CONNECTION_THROUGH_SKIP",
                                        "ONTRIP_TRAIN_START"};
  return strs[static_cast<std::underlying_type_t<state>>(s)];
}

template <typename Label>
inline node* get_node(Label const& l) {
  return l.edge_->to_;
}

template <typename Label>
state next_state(state const s, Label const* c, Label const* n) {
  trace("  next state: {}, {}\n", c->edge_->type_str(),
        (n == nullptr ? "NULL" : n->edge_->type_str()));
  switch (s) {
    case state::AT_STATION:
      if (get_node(*c)->is_route_node()) {
        return state::IN_CONNECTION;
      } else if (n && n->edge_->type() == edge::EXIT_EDGE) {
        return state::PRE_CONNECTION;
      } else if (n && get_node(*n)->is_station_node()) {
        return state::WALK;
      } else {
        return state::PRE_CONNECTION;
      }
    case state::PRE_CONNECTION:
    case state::IN_CONNECTION_THROUGH: return state::IN_CONNECTION;
    case state::IN_CONNECTION_THROUGH_SKIP:
      return get_node(*c)->is_foot_node() ? state::WALK : state::AT_STATION;
    case state::IN_CONNECTION:
      if (c->connection_ == nullptr) {
        if (n && get_node(*c)->type() == node_type::STATION_NODE &&
            get_node(*n)->type() == node_type::FOOT_NODE) {
          if (get_node(*c)->get_station() == get_node(*n)->get_station()) {
            return state::WALK_SKIP;
          } else {
            return state::BWD_WALK;
          }
        }

        switch (get_node(*c)->type()) {
          case node_type::STATION_NODE:
            return n && get_node(*n)->is_station_node() ? state::WALK
                                                        : state::AT_STATION;
          case node_type::FOOT_NODE: return state::WALK;
          case node_type::ROUTE_NODE:
            return get_node(*n)->is_route_node()
                       ? state::IN_CONNECTION_THROUGH
                       : state::IN_CONNECTION_THROUGH_SKIP;
          case node_type::PLATFORM_NODE: return state::AT_STATION;
        }
      } else {
        return state::IN_CONNECTION;
      }
      break;
    case state::WALK:
      if (n && get_node(*n)->is_foot_node()) {
        if (get_node(*c)->get_station() == get_node(*n)->get_station()) {
          return state::WALK_SKIP;
        } else {
          return state::BWD_WALK;
        }
      } else if (n && get_node(*n)->is_station_node()) {
        return state::WALK;
      } else {
        return state::AT_STATION;
      }
    case state::BWD_WALK:
      if (n && get_node(*n)->is_station_node()) {
        return state::BWD_WALK_SKIP;
      } else {
        return state::AT_STATION;
      }
    case state::BWD_WALK_SKIP:
      if ((n && get_node(*n)->type() == node_type::ROUTE_NODE) ||
          (!n && get_node(*c)->type() == node_type::STATION_NODE)) {
        return state::AT_STATION;
      }
    case state::WALK_SKIP:
      if (n && get_node(*c)->type() == node_type::STATION_NODE &&
          get_node(*n)->type() == node_type::FOOT_NODE &&
          get_node(*c)->get_station() != get_node(*n)->get_station()) {
        return state::BWD_WALK;
      } else {
        return state::WALK;
      }
    case state::ONTRIP_TRAIN_START: return state::IN_CONNECTION;
  }
  return static_cast<state>(s);
};

template <typename LabelIt>
state initial_state(LabelIt& it) {
  if (get_node(*it)->is_route_node()) {
    if (get_node(*std::next(it))->is_station_node() ||
        get_node(*std::next(it))->is_platform_node()) {
      ++it;
      return state::AT_STATION;
    } else if (get_node(*std::next(it))->is_foot_node()) {
      ++it;
      return state::WALK;
    } else {
      return state::ONTRIP_TRAIN_START;
    }
  } else if (get_node(*std::next(it))->is_station_node()) {
    return state::WALK;
  } else if (get_node(*std::next(it))->is_foot_node()) {
    if (get_node(*it)->get_station() ==
        get_node(*std::next(it))->get_station()) {
      ++it;
    }
    return state::WALK;
  } else if (auto first_edge_type = std::next(it)->edge_->type();
             first_edge_type == edge::AFTER_TRAIN_BWD_EDGE ||
             first_edge_type == edge::AFTER_TRAIN_FWD_EDGE) {
    // After train foot edge:
    // route node at destination station --AFTER_TRAIN_FOOT_EDGE--> END
    return state::WALK;
  } else {
    return state::AT_STATION;
  }
}

template <typename Label>
std::pair<std::vector<intermediate::stop>, std::vector<intermediate::transport>>
parse_label_chain(schedule const& sched, Label* terminal_label,
                  search_dir const dir) {
  if (kTracing) {
    terminal_label->print(sched, std::cout);
  }

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
  auto current_state = initial_state(it);
  while (it != end(labels)) {
    auto& current = *it;

    trace(
        "state={}, node_type={}, station={}, edge_type={}, mumo_id={}, "
        "connection={}, now={}\n",
        state_to_str(current_state), get_node(current)->type_str(),
        sched.stations_.at(get_node(current)->get_station()->id_)->name_,
        current.edge_->type_str(), current.edge_->get_mumo_id(),
        (current.connection_ == nullptr
             ? "nullptr"
             : get_service_name(sched,
                                current.connection_->full_con_->con_info_)),
        format_time(current.now_));

    switch (current_state) {
      case state::ONTRIP_TRAIN_START:
      case state::AT_STATION: {
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

        auto inc = current_state == state::AT_STATION ? 1 : 0;
        auto s1 = std::next(it, 0 + inc);
        if (s1 != end(labels)) {
          auto s2 = std::next(it, 1 + inc);

          if (current_state == state::ONTRIP_TRAIN_START && s2 != end(labels) &&
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

          // Special case: intermodal AFTER_TRAIN_BWD_EDGE:
          // END (STATION_NODE) << AFTER_TRAIN_EDGE << ROUTE_NODE << connection
          // it                 <<                  << s1         << connection
          if (current.edge_->type() == edge::AFTER_TRAIN_BWD_EDGE &&
              s1 != end(labels) && s1->connection_ != nullptr) {
            trace("  taking d_time from direct predecessor\n");
            auto const& succ = *s1;
            d_track = succ.connection_->full_con_->d_track_;
            d_time = succ.connection_->d_time_;

            auto d_di = get_delay_info(sched, get_node(*it), succ.connection_,
                                       event_type::DEP);
            d_sched_time = d_di.get_schedule_time();
            d_reason = d_di.get_reason();
            d_sched_track = get_schedule_track(
                sched, get_node(*it), succ.connection_, event_type::DEP);
          }
        }

        trace(
            "  Add stop [AT_STATION]: idx={}, id={}, name={}\n", stop_index + 1,
            sched.stations_.at(get_node(current)->get_station()->id_)->eva_nr_,
            sched.stations_.at(get_node(current)->get_station()->id_)->name_);

        stops.emplace_back(
            static_cast<unsigned int>(++stop_index),
            get_node(current)->get_station()->id_, a_track, d_track,
            a_sched_track, d_sched_track, a_time, d_time, a_sched_time,
            d_sched_time, a_reason, d_reason,
            last_con != nullptr && a_time != INVALID_TIME &&
                current.edge_->type() != edge::AFTER_TRAIN_FWD_EDGE,
            d_time != INVALID_TIME);
        break;
      }

      case state::BWD_WALK:
      case state::WALK: {
        utl::verify(std::next(it) != end(labels),
                    "label chain parser in state walk at last label");

        if (last_con != nullptr) {
          walk_arrival_di =
              get_delay_info(sched, last_route_node, last_con, event_type::ARR);
        }

        trace(
            "  Add stop [WALK]: idx={}, id={}, name={}\n", stop_index + 1,
            sched.stations_.at(get_node(current)->get_station()->id_)->eva_nr_,
            sched.stations_.at(get_node(current)->get_station()->id_)->name_);

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

      case state::IN_CONNECTION: {
        if (current.connection_) {
          transports.emplace_back(static_cast<unsigned int>(stop_index),
                                  static_cast<unsigned int>(stop_index) + 1,
                                  current.connection_);
        }

        // do not collect the last connection route node.
        assert(std::next(it) != end(labels));
        auto succ = *std::next(it);
        auto const is_intermodal_dest =
            (succ.edge_->type() == edge::AFTER_TRAIN_FWD_EDGE ||
             succ.edge_->type() == edge::AFTER_TRAIN_BWD_EDGE) &&
            succ.edge_->get_destination(dir)->id_ == 1U;

        if (get_node(succ)->is_route_node() || is_intermodal_dest) {
          auto dep_route_node = get_node(current);

          // skip through edge.
          if (!succ.connection_ && !is_intermodal_dest) {
            dep_route_node = get_node(succ);
            succ = *std::next(it, 2);
          }

          // through edge used but not the route edge after that
          // (instead: went to station node using the leaving edge)
          if ((succ.connection_ || is_intermodal_dest) && current.connection_) {
            auto const a_di =
                current.connection_ == nullptr
                    ? delay_info{}
                    : get_delay_info(sched, get_node(current),
                                     current.connection_, event_type::ARR);
            auto const d_di =
                succ.connection_ == nullptr
                    ? delay_info{}
                    : get_delay_info(sched, dep_route_node, succ.connection_,
                                     event_type::DEP);
            auto const a_sched_track =
                current.connection_ == nullptr
                    ? MOTIS_UNKNOWN_TRACK
                    : get_schedule_track(sched, get_node(current),
                                         current.connection_, event_type::ARR);
            auto const d_sched_track =
                succ.connection_ == nullptr
                    ? MOTIS_UNKNOWN_TRACK
                    : get_schedule_track(sched, dep_route_node,
                                         succ.connection_, event_type::DEP);

            trace(
                "  Add stop [CONNECTION]: idx={}, id={}, name={}, "
                "current.connection={} [{} - {}], succ.connection={} [{} - "
                "{}]\n",
                stop_index + 1,
                sched.stations_.at(get_node(current)->get_station()->id_)
                    ->eva_nr_,
                sched.stations_.at(get_node(current)->get_station()->id_)
                    ->name_,
                current.connection_ != nullptr,
                current.connection_ == nullptr
                    ? format_time(INVALID_TIME)
                    : format_time(current.connection_->d_time_),
                current.connection_ == nullptr
                    ? format_time(INVALID_TIME)
                    : format_time(current.connection_->a_time_),
                succ.connection_ != nullptr,
                succ.connection_ == nullptr
                    ? format_time(INVALID_TIME)
                    : format_time(succ.connection_->d_time_),
                succ.connection_ == nullptr
                    ? format_time(INVALID_TIME)
                    : format_time(succ.connection_->a_time_));
            stops.emplace_back(
                static_cast<unsigned int>(++stop_index),
                get_node(current)->get_station()->id_,
                current.connection_->full_con_->a_track_,  // NOLINT
                succ.connection_ == nullptr
                    ? MOTIS_UNKNOWN_TRACK
                    : succ.connection_->full_con_->d_track_,
                a_sched_track, d_sched_track,
                current.connection_ == nullptr ? MOTIS_UNKNOWN_TRACK
                                               : current.connection_->a_time_,
                succ.connection_ == nullptr ? current.connection_->a_time_
                                            : succ.connection_->d_time_,
                a_di.get_schedule_time(),
                d_di.ev_.is_not_null() ? d_di.get_schedule_time()
                                       : a_di.get_schedule_time(),
                a_di.get_reason(),
                d_di.ev_.is_not_null() ? d_di.get_reason() : a_di.get_reason(),
                dir == search_dir::FWD && is_intermodal_dest,
                dir == search_dir::BWD && is_intermodal_dest);
          } else {
            trace("  IN_CONNECTION: no stop [succ.connection_=null]\n");
          }
        } else {
          trace(
              "  IN_CONNECTION: no stop [succ is not route node: {}, "
              "succ.connection={}, is_intermodal_dest={}\n",
              get_node(succ)->type_str(), fmt::ptr(succ.connection_),
              is_intermodal_dest);
        }

        if (is_intermodal_dest) {
          trace("  Intermodal destination: adding walk {} - {}\n",
                format_time(std::next(it)->now_), format_time(current.now_));
          transports.emplace_back(
              stop_index, static_cast<unsigned int>(stop_index) + 1,
              std::next(it)->now_ - current.now_,
              std::next(it)->edge_->get_mumo_id(), 0,
              std::next(it)->edge_->get_minimum_cost().accessibility_);
          walk_arrival = std::next(it)->now_;
        }

        last_route_node = get_node(current);
        last_con = current.connection_;
        break;
      }

      default:;
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
