#include "motis/csa/csa_to_journey.h"

#include "motis/core/journey/journey_util.h"

#include "motis/routing/output/stop.h"
#include "motis/routing/output/to_journey.h"
#include "motis/routing/output/transport.h"

using namespace motis::routing::output;

namespace motis::csa {

std::pair<std::vector<intermediate::stop>, std::vector<intermediate::transport>>
parse_csa_journey(csa_journey const& csa) {
  std::vector<intermediate::stop> stops;
  std::vector<intermediate::transport> transports;

  if (csa.edges_.empty()) {
    return {stops, transports};
  }

  unsigned stop_idx = 0;
  unsigned a_track = 0;
  time a_time = INVALID_TIME;
  bool exit = false;

  for (auto const& e : csa.edges_) {
    auto const d_track = e.is_connection() ? e.con_->full_con_->d_track_ : 0;
    auto const d_time = e.departure_;
    stops.emplace_back(stop_idx++, e.from_->station_ptr_->index_, a_track,
                       d_track, a_track, d_track, a_time, d_time, a_time,
                       d_time, timestamp_reason::SCHEDULE,
                       timestamp_reason::SCHEDULE, exit, e.enter_);
    a_time = e.arrival_;
    a_track = e.is_connection() ? e.con_->full_con_->a_track_ : 0;
    exit = e.exit_;
    if (e.is_connection()) {
      transports.emplace_back(stop_idx - 1, stop_idx, e.con_);
    } else {
      transports.emplace_back(stop_idx - 1, stop_idx, e.arrival_ - e.departure_,
                              e.mumo_id_, e.mumo_price_, e.mumo_accessibility_);
    }
  }

  auto const& last_edge = csa.edges_.back();
  stops.emplace_back(stop_idx++, last_edge.to_->station_ptr_->index_, a_track,
                     0, a_track, 0, a_time, INVALID_TIME, a_time, INVALID_TIME,
                     timestamp_reason::SCHEDULE, timestamp_reason::SCHEDULE,
                     exit, false);

  return {stops, transports};
};

journey csa_to_journey(schedule const& sched, csa_journey const& csa) {
  auto const parsed = parse_csa_journey(csa);
  auto const& stops = parsed.first;
  auto const& transports = parsed.second;

  journey j;
  j.stops_ = generate_journey_stops(stops, sched);
  j.transports_ = generate_journey_transports(transports, sched);
  j.trips_ = generate_journey_trips(transports, sched);
  j.attributes_ = generate_journey_attributes(transports);
  j.duration_ = csa.duration_;
  j.transfers_ = csa.transfers_;
  j.accessibility_ = csa.accessibility_;
  j.db_costs_ = 0;
  j.price_ = csa.price_;
  j.night_penalty_ = 0;

  return j;
}

}  // namespace motis::csa
