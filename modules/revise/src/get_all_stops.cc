#include "motis/revise/get_all_stops.h"

#include <algorithm>
#include <fstream>
#include <map>
#include <optional>
#include <queue>
#include <set>

#include "boost/range/adaptor/reversed.hpp"

#include "utl/verify.h"

#include "motis/core/common/unixtime.h"
#include "motis/core/schedule/event.h"
#include "motis/core/access/edge_access.h"
#include "motis/core/access/realtime_access.h"
#include "motis/core/access/station_access.h"
#include "motis/core/access/time_access.h"
#include "motis/core/journey/journeys_to_message.h"

#include "motis/revise/get_ev_key.h"

namespace motis::revise {

struct trip_stop : public stop {
  trip_stop(schedule const& sched, ev_key const& arr, ev_key const& dep,
            bool const arr_valid, bool const dep_valid,
            journey::stop const& old_stop)
      : stop(old_stop, arr_valid, dep_valid), dep_(dep), arr_(arr) {
    trip_stop::set_arr_times(sched, arr);
    trip_stop::set_dep_times(sched, dep);
  }

  void set_dep_times(schedule const& sched, ev_key const& k) override {
    if (k) {
      dep_ = k;
      dep_time_ = k.get_time();
      dep_sched_time_ = get_schedule_time(sched, k);
    } else {
      dep_ = ev_key{};
      dep_time_ = arr_time_;
      dep_sched_time_ = arr_sched_time_;
    }
  }

  void set_arr_times(schedule const& sched, ev_key const& k) override {
    if (k) {
      arr_ = k;
      arr_time_ = k.get_time();
      arr_sched_time_ = get_schedule_time(sched, k);
    } else {
      arr_ = ev_key{};
      arr_time_ = 0;
      arr_sched_time_ = 0;
    }
  }

  journey::stop get_stop(schedule const& sched) const override {
    auto const get_ev_info = [&](ev_key const& ev, bool const valid,
                                 bool const is_arr) {
      auto const di = get_delay_info(sched, ev);
      if (ev) {
        return journey::stop::event_info{
            is_arr ? arr_valid_ : dep_valid_,
            motis_to_unixtime(sched, is_arr ? arr_time_ : dep_time_),
            motis_to_unixtime(sched,
                              is_arr ? arr_sched_time_ : dep_sched_time_),
            di.get_reason(),
            sched.tracks_.at(ev.get_track()).str(),
            sched.tracks_.at(get_schedule_track(sched, ev)).str()};
      } else {
        if (valid) {
          auto const type = is_arr ? old_stop_.arrival_ : old_stop_.departure_;
          return journey::stop::event_info{is_arr ? arr_valid_ : dep_valid_,
                                           type.timestamp_,
                                           type.schedule_timestamp_,
                                           type.timestamp_reason_,
                                           type.track_,
                                           type.schedule_track_};
        } else {
          auto const type = is_arr ? old_stop_.arrival_ : old_stop_.departure_;
          return journey::stop::event_info{
              is_arr ? arr_valid_ : dep_valid_,
              motis_to_unixtime(sched, is_arr ? arr_time_ : dep_time_),
              motis_to_unixtime(sched,
                                is_arr ? arr_sched_time_ : dep_sched_time_),
              type.timestamp_reason_,
              type.track_,
              type.schedule_track_};
        }
      }
    };
    auto arr_event_info = get_ev_info(arr_, arr_valid_, true);
    if (propagated_arr_reason_) {
      arr_event_info.timestamp_reason_ = propagated_arr_reason_.value();
    }
    auto dep_event_info = get_ev_info(dep_, dep_valid_, false);
    if (propagated_dep_reason_) {
      dep_event_info.timestamp_reason_ = propagated_dep_reason_.value();
    }
    if (arr_ || dep_) {
      auto const id = arr_ != ev_key{}
                          ? arr_.route_edge_->to_->get_station()->id_
                          : dep_.route_edge_->from_->get_station()->id_;
      auto const station = *sched.stations_[id];
      return journey::stop{exit_,
                           enter_,
                           station.name_.str(),
                           station.eva_nr_.str(),
                           station.width_,
                           station.length_,
                           arr_event_info,
                           dep_event_info};
    } else {
      return journey::stop{
          exit_,          enter_,         old_stop_.name_, old_stop_.eva_no_,
          old_stop_.lat_, old_stop_.lng_, arr_event_info,  dep_event_info};
    }
  }

  void propagate_time(schedule const&, stop const& pred) override {
    if (pred.get_type() == stop_type_t::WALK_STOP) {
      arr_sched_time_ = pred.dep_sched_time_;
      arr_time_ = pred.dep_time_;
    }
  }

  void propagate_time_bwd(schedule const&, stop&) override {}

  timestamp_reason get_timestamp_reason(schedule const& sched,
                                        event_type const type) const override {
    auto const ev = type == event_type::ARR ? arr_ : dep_;
    if (ev) {
      return get_delay_info(sched, ev).get_reason();
    } else {
      return type == event_type::ARR ? old_stop_.arrival_.timestamp_reason_
                                     : old_stop_.departure_.timestamp_reason_;
    }
  }

  void set_timestamp_reason(event_type const type,
                            timestamp_reason const reason) override {
    if (type == event_type::ARR) {
      propagated_arr_reason_ = reason;
    } else {
      propagated_dep_reason_ = reason;
    }
  }

  stop_type_t get_type() const override { return stop_type_t::TRIP_STOP; };
  ev_key get_arr() const override { return arr_; }
  ev_key get_dep() const override { return dep_; }
  void set_arr(ev_key const& arr) override { arr_ = arr; }
  void set_dep(ev_key const& dep) override { dep_ = dep; }

  ev_key dep_, arr_;
  std::optional<timestamp_reason> propagated_arr_reason_;
  std::optional<timestamp_reason> propagated_dep_reason_;
};

struct walk_stop : public stop {
  walk_stop(schedule const& sched, int walk_time, journey::stop const& old_stop)
      : stop(old_stop, old_stop.arrival_.valid_, old_stop.departure_.valid_),
        walk_time_(walk_time) {
    walk_stop::set_arr_times(sched, ev_key{});
    walk_stop::set_dep_times(sched, ev_key{});
  }

  void set_dep_times(schedule const& sched, ev_key const&) override {
    if (dep_valid_) {
      dep_time_ =
          unix_to_motistime(sched, old_stop_.departure_.schedule_timestamp_);
      dep_sched_time_ = dep_time_;
    }
  }

  void set_arr_times(schedule const& sched, ev_key const&) override {
    if (arr_valid_) {
      arr_time_ =
          unix_to_motistime(sched, old_stop_.arrival_.schedule_timestamp_);
      arr_sched_time_ = arr_time_;
    }
  }

  journey::stop get_stop(schedule const& sched) const override {
    auto journey_stop = old_stop_;
    journey_stop.exit_ = exit_;
    journey_stop.enter_ = enter_;
    if (arr_valid_) {
      journey_stop.arrival_.timestamp_ = motis_to_unixtime(sched, arr_time_);
      journey_stop.arrival_.schedule_timestamp_ =
          motis_to_unixtime(sched, arr_sched_time_);
      journey_stop.arrival_.timestamp_reason_ = timestamp_reason_;
    }
    if (dep_valid_) {
      journey_stop.departure_.timestamp_ = motis_to_unixtime(sched, dep_time_);
      journey_stop.departure_.schedule_timestamp_ =
          motis_to_unixtime(sched, dep_sched_time_);
      journey_stop.departure_.timestamp_reason_ = timestamp_reason_;
    }
    return journey_stop;
  }

  void propagate_time(schedule const& sched, stop const& pred) override {
    if (pred.get_type() == stop_type_t::TRIP_STOP) {
      dep_time_ =
          pred.dep_valid_
              ? pred.dep_time_
              : unix_to_motistime(sched, pred.old_stop_.departure_.timestamp_);
      dep_sched_time_ =
          pred.dep_valid_
              ? pred.dep_sched_time_
              : unix_to_motistime(
                    sched, pred.old_stop_.departure_.schedule_timestamp_);
      arr_time_ =
          pred.arr_valid_
              ? pred.arr_time_
              : unix_to_motistime(sched, pred.old_stop_.arrival_.timestamp_);
      arr_sched_time_ =
          pred.arr_valid_
              ? pred.arr_sched_time_
              : unix_to_motistime(sched,
                                  pred.old_stop_.arrival_.schedule_timestamp_);
    } else {
      arr_time_ = static_cast<time>(pred.dep_time_ + walk_time_);
      arr_sched_time_ = static_cast<time>(pred.dep_sched_time_ + walk_time_);
      if (dep_valid_) {
        dep_time_ = arr_time_;
        dep_sched_time_ = arr_sched_time_;
      } else {
        dep_time_ = INVALID_TIME;
        dep_sched_time_ = INVALID_TIME;
      }
    }
    timestamp_reason_ = pred.get_timestamp_reason(sched, event_type::ARR);
  }

  void propagate_time_bwd(schedule const& sched, stop& next) override {
    if (next.get_type() == stop_type_t::TRIP_STOP) {
      dep_time_ =
          next.dep_valid_
              ? next.dep_time_
              : unix_to_motistime(sched, next.old_stop_.departure_.timestamp_);
      dep_sched_time_ =
          next.dep_valid_
              ? next.dep_sched_time_
              : unix_to_motistime(
                    sched, next.old_stop_.departure_.schedule_timestamp_);
      arr_time_ = dep_time_;
      arr_sched_time_ = dep_sched_time_;
      next.arr_time_ = arr_time_;
      next.arr_sched_time_ = arr_sched_time_;
      if (timestamp_reason_ == timestamp_reason::SCHEDULE) {
        next.set_timestamp_reason(
            event_type::ARR, next.get_timestamp_reason(sched, event_type::DEP));
      }
    } else {
      auto const& next_walk = dynamic_cast<walk_stop const&>(next);
      dep_time_ = static_cast<time>(next.arr_time_ - next_walk.walk_time_);
      dep_sched_time_ =
          static_cast<time>(next.arr_sched_time_ - next_walk.walk_time_);
      if (arr_valid_) {
        arr_time_ = dep_time_;
        arr_sched_time_ = dep_sched_time_;
      } else {
        arr_time_ = INVALID_TIME;
        arr_sched_time_ = INVALID_TIME;
      }
    }
    if (timestamp_reason_ == timestamp_reason::SCHEDULE) {
      timestamp_reason_ = next.get_timestamp_reason(sched, event_type::DEP);
    }
  }

  timestamp_reason get_timestamp_reason(schedule const&,
                                        event_type const) const override {
    return timestamp_reason_;
  }

  void set_timestamp_reason(event_type const,
                            timestamp_reason const reason) override {
    timestamp_reason_ = reason;
  }

  stop_type_t get_type() const override { return stop_type_t::WALK_STOP; };
  ev_key get_arr() const override { return ev_key{}; }
  ev_key get_dep() const override { return ev_key{}; }
  void set_arr(ev_key const&) override {}
  void set_dep(ev_key const&) override {}

  int walk_time_;
  timestamp_reason timestamp_reason_{timestamp_reason::SCHEDULE};
};

std::vector<ev_key> get_shortest_path(schedule const& sched, journey const& j,
                                      int const from, int const to) {
  auto const tgt = get_station_node(sched, j.stops_[to].eva_no_);
  auto const start_k = get_ev_key(sched, j, from, event_type::DEP);
  if (!start_k) {
    return {};
  }

  std::set<trip::route_edge> visited;
  std::queue<trip::route_edge> q;
  std::map<trip::route_edge, trip::route_edge> pred;

  visited.insert(start_k.route_edge_);
  q.push(start_k.route_edge_);

  while (!q.empty()) {
    auto const entry = q.front();
    q.pop();
    auto const e = entry.get_edge();

    if (e->to_->get_station() == tgt) {
      std::vector<ev_key> path;
      auto re = &entry;
      while (re != nullptr) {
        if (re->get_edge()->type() == edge::ROUTE_EDGE) {
          path.push_back(ev_key{*re, start_k.lcon_idx_, event_type::DEP});
        }
        auto const p = pred.find(*re);
        re = p == end(pred) ? nullptr : &(p->second);
      }
      std::reverse(begin(path), end(path));
      return path;
    }

    for (auto const& out : e->to_->edges_) {
      if (out.type() != edge::THROUGH_EDGE && out.type() != edge::ROUTE_EDGE) {
        continue;
      }

      trip::route_edge out_re{&out};

      if (visited.insert(out_re).second) {
        q.push(out_re);
        pred[out_re] = entry;
      }
    }
  }

  return {};
}

std::vector<section> get_sections(journey const& j) {
  // Get first enter and last exit.
  auto const first_it =
      std::find_if(begin(j.stops_), end(j.stops_),
                   [](journey::stop const& s) { return s.enter_; });
  auto const last_it =
      std::find_if(j.stops_.rbegin(), j.stops_.rend(),
                   [](journey::stop const& s) { return s.exit_; });
  utl::verify(first_it != end(j.stops_),
              "get sections(first) : invalid journey");
  utl::verify(last_it != j.stops_.rend(),
              "get sections(last) : invalid journey");

  // Get sections.
  std::vector<section> sections;
  sections.emplace_back(
      0, static_cast<int>(std::distance(begin(j.stops_), first_it)),
      section_type::WALK);
  for (auto it = first_it; it != last_it.base(); ++it) {
    auto const distance =
        static_cast<int>(std::distance(std::begin(j.stops_), it));
    if (it->exit_) {
      sections.back().to_ = distance;
      sections.emplace_back(distance, -1, section_type::WALK);
    }
    if (it->enter_) {
      sections.back().to_ = distance;
      sections.emplace_back(distance, -1, section_type::TRIP);
    }
  }
  sections.back().to_ = j.stops_.size() - 1;
  utl::verify(std::all_of(begin(sections), end(sections),
                          [](section const& s) {
                            return s.from_ != -1 && s.to_ != -1;
                          }),
              "get secctions(all) : invalid journey");
  return sections;
}

std::vector<stop_ptr> get_trip_stops(schedule const& sched, journey const& j,
                                     int from, int to) {
  std::vector<stop_ptr> stops;
  for (auto const& k : get_shortest_path(sched, j, from, to)) {
    if (stops.empty()) {
      stops.emplace_back(std::make_unique<trip_stop>(sched, ev_key{}, k, true,
                                                     true, journey::stop{}));
      stops.back()->enter_ = true;
    } else {
      stops.back()->set_dep_times(sched, k);
    }
    stops.emplace_back(std::make_unique<trip_stop>(
        sched, k.get_opposite(), ev_key{}, true, true, journey::stop{}));
  }
  if (stops.empty()) {
    for (unsigned i = from; i <= static_cast<unsigned>(to); ++i) {
      stops.emplace_back(std::make_unique<trip_stop>(
          sched, ev_key{}, ev_key{}, false, false, j.stops_.at(i)));
    }
    stops.front()->enter_ = true;
  }
  stops.back()->exit_ = true;
  return stops;
}

std::vector<stop_ptr> get_walk_stops(schedule const& sched, journey const& j,
                                     int from, int to) {
  std::vector<stop_ptr> stops;
  utl::verify(from < to, "get walk stop : invalid journey");
  for (auto i = from; i <= to; i++) {
    if (i == from) {
      stops.emplace_back(std::make_unique<walk_stop>(sched, 0, j.stops_.at(i)));
    } else {
      stops.emplace_back(std::make_unique<walk_stop>(
          sched,
          static_cast<int>((j.stops_.at(i).arrival_.schedule_timestamp_ -
                            j.stops_.at(i - 1).departure_.schedule_timestamp_) /
                           60),
          j.stops_.at(i)));
    }
  }
  return stops;
}

int get_transfer_time(schedule const& sched, stop const* s, ev_key const& e) {
  if (e.is_not_null()) {
    return sched.stations_[e.get_station_idx()]->transfer_time_;
  } else if (!s->old_stop_.eva_no_.empty()) {
    return get_station(sched, s->old_stop_.eva_no_)->transfer_time_;
  } else {
    return 0;
  }
}

void add_initial_interchange_time(schedule const& sched,
                                  std::vector<stop_ptr>& stops) {
  if (stops.empty() || stops.front()->get_type() != stop_type_t::WALK_STOP) {
    return;
  }
  auto const first_trip_stop =
      std::find_if(begin(stops), end(stops), [](stop_ptr const& s) {
        return s->get_type() == stop_type_t::TRIP_STOP;
      });
  if (first_trip_stop == begin(stops) || first_trip_stop == end(stops)) {
    return;
  }
  auto const interchange_time = get_transfer_time(
      sched, first_trip_stop->get(), first_trip_stop->get()->get_dep());
  for (auto& s : stops) {
    if (s->get_type() == stop_type_t::TRIP_STOP) {
      s->arr_time_ -= interchange_time;
      s->arr_sched_time_ -= interchange_time;
      break;
    }
    s->dep_time_ -= interchange_time;
    s->dep_sched_time_ -= interchange_time;
    if (s->arr_time_ != INVALID_TIME) {
      s->arr_time_ -= interchange_time;
      s->arr_sched_time_ -= interchange_time;
    }
  }
}

void add_final_interchange_time(schedule const& sched,
                                std::vector<stop_ptr>& stops) {
  if (stops.empty() || stops.back()->get_type() != stop_type_t::WALK_STOP) {
    return;
  }
  auto const last_trip_stop =
      std::find_if(stops.crbegin(), stops.crend(), [](stop_ptr const& s) {
        return s->get_type() == stop_type_t::TRIP_STOP;
      });
  if (last_trip_stop == stops.crend() ||
      std::distance(stops.crbegin(), last_trip_stop) > 1) {
    return;
  }
  auto const interchange_time = get_transfer_time(
      sched, last_trip_stop->get(), last_trip_stop->get()->get_arr());
  for (auto& s : boost::adaptors::reverse(stops)) {
    if (s->get_type() == stop_type_t::TRIP_STOP) {
      s->dep_time_ += interchange_time;
      s->dep_sched_time_ += interchange_time;
      break;
    }
    s->arr_time_ += interchange_time;
    s->arr_sched_time_ += interchange_time;
    if (s->dep_time_ != INVALID_TIME) {
      s->dep_time_ += interchange_time;
      s->dep_sched_time_ += interchange_time;
    }
  }
}

void propagate_times(schedule const& sched, std::vector<stop_ptr>& stops) {
  stop* pred = nullptr;
  for (auto const& s : stops) {
    if (pred != nullptr) {
      s->propagate_time(sched, *pred);
    }
    pred = s.get();
  }
  auto const first_trip_stop =
      std::find_if(begin(stops), end(stops), [](stop_ptr const& s) {
        return s->get_type() == stop_type_t::TRIP_STOP;
      });
  if (first_trip_stop == begin(stops) || first_trip_stop == end(stops)) {
    return;
  }
  for (auto it = std::prev(first_trip_stop); it >= begin(stops);
       it = std::prev(it)) {
    it->get()->propagate_time_bwd(sched, **std::next(it));
  }
}

void generate_stops(schedule const& sched, std::vector<stop_ptr>& stops) {
  stop_ptr deleted_pred;
  for (auto i = 1UL; i < stops.size(); ++i) {
    auto& curr = stops.at(i);
    auto& pred = deleted_pred ? deleted_pred : stops.at(i - 1);

    if (pred->get_type() == stop_type_t::TRIP_STOP &&
        curr->get_type() == stop_type_t::TRIP_STOP && pred->exit_ &&
        curr->enter_) {
      pred->set_dep_times(sched, curr->get_dep());
      pred->set_enter(true);
      pred->dep_valid_ = curr->dep_valid_;
      pred->set_dep(curr->get_dep());
      if (pred->old_stop_.eva_no_.empty()) {
        pred->old_stop_ = curr->old_stop_;
      }
      stops.erase(std::next(begin(stops), i--));
    } else if (pred->get_type() == stop_type_t::TRIP_STOP &&
               curr->get_type() == stop_type_t::WALK_STOP) {
      pred->set_exit(true);
      pred->dep_time_ = pred->arr_time_;
      pred->dep_sched_time_ = pred->arr_sched_time_;
      pred->set_timestamp_reason(
          event_type::DEP, pred->get_timestamp_reason(sched, event_type::ARR));
      deleted_pred = std::move(curr);
      stops.erase(std::next(begin(stops), i--));
      continue;
    } else if (pred->get_type() == stop_type_t::WALK_STOP &&
               curr->get_type() == stop_type_t::TRIP_STOP) {
      curr->set_enter(true);
      curr->arr_time_ = pred->dep_time_;
      curr->arr_sched_time_ = pred->dep_sched_time_;
      stops.erase(std::next(begin(stops), --i));
    }
    deleted_pred.reset();
  }

  if (stops.front()->get_type() == stop_type_t::TRIP_STOP) {
    stops.front()->set_enter(true);
  }
  stops.front()->arr_time_ = INVALID_TIME;
  stops.front()->arr_sched_time_ = INVALID_TIME;

  if (stops.back()->get_type() == stop_type_t::TRIP_STOP) {
    stops.back()->set_exit(true);
  }
  stops.back()->dep_time_ = INVALID_TIME;
  stops.back()->dep_sched_time_ = INVALID_TIME;
}

std::vector<stop_ptr> get_all_stops(schedule const& sched, journey const& j) {
  std::vector<stop_ptr> stops;
  for (auto const& section : get_sections(j)) {
    if (section.from_ == section.to_) {
      continue;
    }
    if (section.type_ == section_type::TRIP) {
      auto trip_stops = get_trip_stops(sched, j, section.from_, section.to_);
      stops.insert(end(stops), std::make_move_iterator(begin(trip_stops)),
                   std::make_move_iterator(trip_stops.end()));

    } else {
      auto walk_stops = get_walk_stops(sched, j, section.from_, section.to_);
      stops.insert(end(stops), std::make_move_iterator(begin(walk_stops)),
                   std::make_move_iterator(walk_stops.end()));
    }
  }

  if (!stops.empty()) {
    stops.front()->arr_valid_ = false;
    stops.back()->dep_valid_ = false;
  }

  propagate_times(sched, stops);

  generate_stops(sched, stops);

  add_initial_interchange_time(sched, stops);
  add_final_interchange_time(sched, stops);

  return stops;
}

}  // namespace motis::revise
