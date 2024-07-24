#include "motis/core/journey/journeys_to_message.h"

#include "boost/date_time/gregorian/gregorian_types.hpp"
#include "boost/date_time/posix_time/posix_time.hpp"

#include "motis/core/conv/connection_status_conv.h"
#include "motis/core/conv/problem_type_conv.h"
#include "motis/core/conv/trip_conv.h"

using namespace flatbuffers;
using namespace motis::module;
using namespace motis::routing;

namespace motis {

TimestampReason convert_reason(timestamp_reason const r) {
  switch (r) {
    case timestamp_reason::SCHEDULE: return TimestampReason_SCHEDULE;
    case timestamp_reason::IS: return TimestampReason_IS;
    case timestamp_reason::FORECAST: return TimestampReason_FORECAST;
    case timestamp_reason::PROPAGATION: return TimestampReason_PROPAGATION;
    default: return TimestampReason_SCHEDULE;
  }
}

std::vector<Offset<Stop>> convert_stops(FlatBufferBuilder& b,
                                        std::vector<journey::stop> const& stops,
                                        bool const include_invalid_stop_info) {
  std::vector<Offset<Stop>> buf_stops;

  for (auto const& stop : stops) {
    auto const arr =
        stop.arrival_.valid_ || include_invalid_stop_info
            ? CreateEventInfo(b, stop.arrival_.timestamp_,
                              stop.arrival_.schedule_timestamp_,
                              b.CreateString(stop.arrival_.track_),
                              b.CreateString(stop.arrival_.schedule_track_),
                              stop.arrival_.valid_,
                              convert_reason(stop.arrival_.timestamp_reason_))
            : CreateEventInfo(b, 0, 0, b.CreateString(""), b.CreateString(""),
                              stop.arrival_.valid_, TimestampReason_SCHEDULE);
    auto const dep =
        stop.departure_.valid_ || include_invalid_stop_info
            ? CreateEventInfo(b, stop.departure_.timestamp_,
                              stop.departure_.schedule_timestamp_,
                              b.CreateString(stop.departure_.track_),
                              b.CreateString(stop.departure_.schedule_track_),
                              stop.departure_.valid_,
                              convert_reason(stop.departure_.timestamp_reason_))
            : CreateEventInfo(b, 0, 0, b.CreateString(""), b.CreateString(""),
                              stop.departure_.valid_, TimestampReason_SCHEDULE);
    auto const pos = Position(stop.lat_, stop.lng_);
    buf_stops.push_back(
        CreateStop(b,
                   CreateStation(b, b.CreateString(stop.eva_no_),
                                 b.CreateString(stop.name_), &pos),
                   arr, dep, static_cast<uint8_t>(stop.exit_) != 0U,
                   static_cast<uint8_t>(stop.enter_) != 0U));
  }

  return buf_stops;
}

std::vector<Offset<MoveWrapper>> convert_moves(
    FlatBufferBuilder& b, std::vector<journey::transport> const& transports) {
  std::vector<Offset<MoveWrapper>> moves;

  auto const convert_color =
      [&b](uint32_t color) -> flatbuffers::Offset<flatbuffers::String> {
    if (color == 0U) {
      return 0;
    }
    return b.CreateString(fmt::format("{:06x}", color & 0x00ffffff));
  };

  for (auto const& t : transports) {
    Range const r(t.from_, t.to_);
    if (t.is_walk_) {
      moves.push_back(CreateMoveWrapper(
          b, Move_Walk,
          CreateWalk(b, &r, t.mumo_id_, t.mumo_price_, t.mumo_accessibility_,
                     b.CreateString(t.mumo_type_))
              .Union()));
    } else {
      moves.push_back(CreateMoveWrapper(
          b, Move_Transport,
          CreateTransport(
              b, &r, t.clasz_, b.CreateString(t.line_identifier_),
              b.CreateString(t.name_), b.CreateString(t.provider_),
              b.CreateString(t.provider_url_), b.CreateString(t.direction_),
              convert_color(t.route_color_), convert_color(t.route_text_color_))
              .Union()));
    }
  }

  return moves;
}

std::vector<Offset<Trip>> convert_trips(
    FlatBufferBuilder& b, std::vector<journey::trip> const& trips) {
  std::vector<Offset<Trip>> journey_trips;

  for (auto const& t : trips) {
    auto const r =
        Range{static_cast<int16_t>(t.from_), static_cast<int16_t>(t.to_)};
    journey_trips.push_back(
        CreateTrip(b, &r, to_fbs(b, t.extern_trip_), b.CreateString(t.debug_)));
  }

  return journey_trips;
}

std::vector<Offset<Attribute>> convert_attributes(
    FlatBufferBuilder& b,
    std::vector<journey::ranged_attribute> const& attributes) {
  std::vector<Offset<Attribute>> buf_attributes;
  for (auto const& a : attributes) {
    auto const r =
        Range{static_cast<int16_t>(a.from_), static_cast<int16_t>(a.to_)};
    buf_attributes.push_back(CreateAttribute(
        b, &r, b.CreateString(a.attr_.code_), b.CreateString(a.attr_.text_)));
  }
  return buf_attributes;
}

std::vector<Offset<FreeText>> convert_free_texts(
    FlatBufferBuilder& b,
    std::vector<journey::ranged_free_text> const& free_texts) {
  std::vector<Offset<FreeText>> buf_free_texts;
  for (auto const& f : free_texts) {
    auto const r =
        Range{static_cast<int16_t>(f.from_), static_cast<int16_t>(f.to_)};
    buf_free_texts.push_back(CreateFreeText(b, &r, f.text_.code_,
                                            b.CreateString(f.text_.text_),
                                            b.CreateString(f.text_.type_)));
  }
  return buf_free_texts;
}

std::vector<Offset<Problem>> convert_problems(
    FlatBufferBuilder& b, std::vector<journey::problem> const& problems) {
  std::vector<Offset<Problem>> buf_problems;
  for (auto const& p : problems) {
    auto const r =
        Range{static_cast<int16_t>(p.from_), static_cast<int16_t>(p.to_)};
    buf_problems.push_back(CreateProblem(b, &r, problem_type_to_fbs(p.type_)));
  }
  return buf_problems;
}

Offset<Connection> to_connection(FlatBufferBuilder& b, journey const& j,
                                 bool const include_invalid_stop_info) {
  return CreateConnection(
      b, b.CreateVector(convert_stops(b, j.stops_, include_invalid_stop_info)),
      b.CreateVector(convert_moves(b, j.transports_)),
      b.CreateVector(convert_trips(b, j.trips_)),
      b.CreateVector(convert_attributes(b, j.attributes_)),
      b.CreateVector(convert_free_texts(b, j.free_texts_)),
      b.CreateVector(convert_problems(b, j.problems_)), j.night_penalty_,
      j.db_costs_, status_to_fbs(j.status_));
}

}  // namespace motis
