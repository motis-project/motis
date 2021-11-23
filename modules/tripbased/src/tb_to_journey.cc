#include "motis/tripbased/tb_to_journey.h"

#include "motis/core/journey/journey_util.h"

#include "motis/core/access/trip_stop.h"
#include "motis/routing/output/stop.h"
#include "motis/routing/output/to_journey.h"
#include "motis/routing/output/transport.h"

using namespace motis::access;
using namespace motis::routing::output;

namespace motis::tripbased {

std::pair<std::vector<intermediate::stop>, std::vector<intermediate::transport>>
parse_tb_journey(schedule const& sched, tb_journey const& tbj) {
  std::vector<intermediate::stop> stops;
  std::vector<intermediate::transport> transports;

  if (tbj.edges_.empty()) {
    return {stops, transports};
  }

  unsigned stop_idx = 0;
  unsigned a_track = 0;
  time a_time = INVALID_TIME;
  bool exit = false;

  for (auto const& e : tbj.edges_) {
    if (e.is_connection()) {
      auto const trp = sched.expanded_trips_.data_[e.trip_];
      assert(trp != nullptr);
      assert(e.to_stop_index_ > e.from_stop_index_);
      for (auto trip_stop_idx = e.from_stop_index_;
           trip_stop_idx < e.to_stop_index_; ++trip_stop_idx) {
        trip_stop stop{trp, trip_stop_idx};
        assert(stop.has_departure());
        auto const& dep_lcon = stop.dep_lcon();
        auto const enter = trip_stop_idx == e.from_stop_index_;
        stops.emplace_back(stop_idx++, stop.get_station_id(), a_track,
                           dep_lcon.full_con_->d_track_, a_track,
                           dep_lcon.full_con_->d_track_, a_time,
                           dep_lcon.d_time_, a_time, dep_lcon.d_time_,
                           timestamp_reason::SCHEDULE,
                           timestamp_reason::SCHEDULE, exit, enter);
        assert(!enter || stop.get_route_node()->is_in_allowed());
        transports.emplace_back(stop_idx - 1, stop_idx, &dep_lcon);
        a_time = dep_lcon.a_time_;
        a_track = dep_lcon.full_con_->a_track_;
        exit = false;
      }
      assert((
          trip_stop{trp, e.to_stop_index_}.get_route_node()->is_out_allowed()));
      exit = true;
    } else {
      auto const d_time = e.departure_time_;
      stops.emplace_back(stop_idx++, e.footpath_.from_stop_, a_track, 0,
                         a_track, 0, a_time, d_time, a_time, d_time,
                         timestamp_reason::SCHEDULE, timestamp_reason::SCHEDULE,
                         exit, false);
      transports.emplace_back(stop_idx - 1, stop_idx, e.footpath_.duration_,
                              e.mumo_id_, e.mumo_price_, e.mumo_accessibility_);
      a_time = static_cast<time>(d_time + e.footpath_.duration_);
      a_track = 0;
      exit = false;
    }
  }

  auto const& last_edge = tbj.edges_.back();
  auto last_stop = 0U;
  if (last_edge.is_connection()) {
    auto const trp = sched.expanded_trips_.data_[last_edge.trip_];
    assert(trp != nullptr);
    trip_stop stop{trp, last_edge.to_stop_index_};
    last_stop = stop.get_station_id();
  } else {
    last_stop = last_edge.footpath_.to_stop_;
  }
  stops.emplace_back(stop_idx++, last_stop, a_track, 0, a_track, 0, a_time,
                     INVALID_TIME, a_time, INVALID_TIME,
                     timestamp_reason::SCHEDULE, timestamp_reason::SCHEDULE,
                     exit, false);

  return {stops, transports};
}

journey tb_to_journey(schedule const& sched, tb_journey const& tbj) {
  assert(tbj.is_reconstructed());
  auto const parsed = parse_tb_journey(sched, tbj);
  auto const& stops = parsed.first;
  auto const& transports = parsed.second;

  journey j;
  j.stops_ = generate_journey_stops(stops, sched);
  j.transports_ = generate_journey_transports(transports, sched);
  j.trips_ = generate_journey_trips(transports, sched);
  j.attributes_ = generate_journey_attributes(transports);
  j.duration_ = tbj.duration_;
  j.transfers_ = tbj.transfers_;
  j.accessibility_ = tbj.accessibility_;
  j.db_costs_ = 0;
  j.night_penalty_ = 0;

  return j;
}

}  // namespace motis::tripbased
