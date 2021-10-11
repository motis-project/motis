#include "motis/core/journey/message_to_journeys.h"

#include <algorithm>

#include "motis/core/schedule/category.h"
#include "motis/core/conv/connection_status_conv.h"
#include "motis/core/conv/problem_type_conv.h"
#include "motis/core/conv/timestamp_reason_conv.h"
#include "motis/core/journey/journey.h"
#include "motis/core/journey/journey_util.h"

#include "motis/protocol/RoutingResponse_generated.h"

namespace motis {

journey::stop::event_info to_event_info(EventInfo const& event) {
  journey::stop::event_info e;
  e.track_ = event.track()->c_str();
  e.schedule_track_ = event.schedule_track()->c_str();
  e.timestamp_ = event.time();
  e.schedule_timestamp_ = event.schedule_time();
  e.timestamp_reason_ = from_fbs(event.reason());
  e.valid_ = event.valid();
  return e;
}

journey::stop to_stop(Stop const& stop) {
  journey::stop s;
  s.eva_no_ = stop.station()->id()->c_str();
  s.exit_ = static_cast<bool>(stop.exit());
  s.enter_ = static_cast<bool>(stop.enter());
  s.lat_ = stop.station()->pos()->lat();
  s.lng_ = stop.station()->pos()->lng();
  s.name_ = stop.station()->name()->c_str();
  s.arrival_ = to_event_info(*stop.arrival());
  s.departure_ = to_event_info(*stop.departure());
  return s;
}

journey::transport to_transport(Walk const& walk, time const duration) {
  auto t = journey::transport{};
  t.is_walk_ = true;
  t.duration_ = duration;
  t.from_ = walk.range()->from();
  t.to_ = walk.range()->to();
  t.mumo_id_ = walk.mumo_id();
  t.mumo_price_ = walk.price();
  t.mumo_accessibility_ = walk.accessibility();
  t.mumo_type_ = walk.mumo_type()->c_str();
  return t;
}

journey::transport to_transport(Transport const& transport,
                                time const duration) {
  auto t = journey::transport{};
  t.duration_ = duration;
  t.from_ = transport.range()->from();
  t.to_ = transport.range()->to();
  t.is_walk_ = false;
  t.category_name_ = transport.category_name()->c_str();
  t.category_id_ = transport.category_id();
  t.clasz_ = transport.clasz();
  t.direction_ = transport.direction()->c_str();
  t.line_identifier_ = transport.line_id()->c_str();
  t.name_ = transport.name()->c_str();
  t.provider_ = transport.provider()->c_str();
  t.mumo_id_ = 0;
  t.train_nr_ = transport.train_nr();
  return t;
}

journey::trip to_trip(Trip const& trip) {
  auto t = journey::trip{};
  t.from_ = trip.range()->from();
  t.to_ = trip.range()->to();
  t.extern_trip_.station_id_ = trip.id()->station_id()->str();
  t.extern_trip_.train_nr_ = trip.id()->train_nr();
  t.extern_trip_.time_ = trip.id()->time();
  t.extern_trip_.target_station_id_ = trip.id()->target_station_id()->str();
  t.extern_trip_.target_time_ = trip.id()->target_time();
  t.extern_trip_.line_id_ = trip.id()->line_id()->str();
  t.debug_ = trip.debug()->str();
  return t;
}

journey::ranged_attribute to_attribute(Attribute const& attribute) {
  journey::ranged_attribute a;
  a.attr_.code_ = attribute.code()->c_str();
  a.attr_.text_ = attribute.text()->c_str();
  a.from_ = attribute.range()->from();
  a.to_ = attribute.range()->to();
  return a;
}

journey::ranged_free_text to_free_text(FreeText const& free_text) {
  journey::ranged_free_text f;
  f.text_.code_ = free_text.code();
  f.text_.text_ = free_text.text()->c_str();
  f.text_.type_ = free_text.type()->c_str();
  f.from_ = free_text.range()->from();
  f.to_ = free_text.range()->to();
  return f;
}

journey::problem to_problem(Problem const& problem) {
  journey::problem p;
  p.type_ = problem_type_from_fbs(problem.type());
  p.from_ = problem.range()->from();
  p.to_ = problem.range()->to();
  return p;
}

time get_move_duration(
    Range const& range,
    flatbuffers::Vector<flatbuffers::Offset<Stop>> const& stops) {
  auto const dep_time = stops[range.from()]->departure()->time();
  auto const arr_time = stops[range.to()]->arrival()->time();
  auto const minutes = static_cast<int64_t>(arr_time - dep_time / 60);
  return time{minutes};
}

journey convert(Connection const* conn) {
  journey journey;

  /* stops */
  for (auto const& stop : *conn->stops()) {
    journey.stops_.push_back(to_stop(*stop));
  }

  /* transports */
  for (auto const& move : *conn->transports()) {
    if (move->move_type() == Move_Walk) {
      auto const walk = reinterpret_cast<Walk const*>(move->move());
      journey.transports_.push_back(to_transport(
          *walk, get_move_duration(*walk->range(), *conn->stops())));
    } else if (move->move_type() == Move_Transport) {
      auto const transport = reinterpret_cast<Transport const*>(move->move());
      journey.transports_.push_back(to_transport(
          *transport, get_move_duration(*transport->range(), *conn->stops())));
    }
  }

  /* attributes */
  for (auto const& attribute : *conn->attributes()) {
    journey.attributes_.push_back(to_attribute(*attribute));
  }

  /* trips */
  for (auto const& trp : *conn->trips()) {
    journey.trips_.push_back(to_trip(*trp));
  }

  /* free_texts */
  for (auto const& f : *conn->free_texts()) {
    journey.free_texts_.push_back(to_free_text(*f));
  }

  /* problems */
  for (auto const& p : *conn->problems()) {
    journey.problems_.push_back(to_problem(*p));
  }

  journey.status_ = status_from_fbs(conn->status());
  journey.duration_ = get_duration(journey);
  journey.transfers_ = get_transfers(journey);
  journey.night_penalty_ = conn->night_penalty();
  journey.db_costs_ = conn->db_costs();
  journey.accessibility_ = get_accessibility(journey);

  return journey;
}

std::vector<journey> message_to_journeys(
    routing::RoutingResponse const* response) {
  std::vector<journey> journeys;
  for (auto conn : *response->connections()) {
    journeys.push_back(convert(conn));
  }
  return journeys;
}

}  // namespace motis
