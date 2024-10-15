#include "motis/ris/risml/risml_parser.h"

#include <map>
#include <optional>

#include "pugixml.hpp"

#include "utl/parser/cstr.h"
#include "utl/parser/mmap_reader.h"
#include "utl/verify.h"

#include "motis/protocol/RISMessage_generated.h"

#include "motis/core/common/logging.h"
#include "motis/core/common/unixtime.h"
#include "motis/ris/risml/common.h"
#include "motis/ris/risml/parse_event.h"
#include "motis/ris/risml/parse_station.h"
#include "motis/ris/risml/parse_time.h"
#include "motis/ris/risml/parse_type.h"

#ifdef CreateEvent
#undef CreateEvent
#endif

using namespace std::placeholders;
using namespace flatbuffers;
using namespace pugi;
using namespace utl;
using namespace motis::logging;

namespace motis::ris::risml {

template <typename F>
void inline foreach_event(
    context& ctx, xml_node const& msg, F func,
    char const* train_selector = "./Service/ListZug/Zug") {
  for (auto const& train : msg.select_nodes(train_selector)) {
    auto const& t_node = train.node();
    auto service_num = t_node.attribute("Nr").as_uint();
    auto line_id = t_node.attribute("Linie").value();
    auto line_id_offset = ctx.b_.CreateString(line_id);

    for (auto const& train_event : t_node.select_nodes("./ListZE/ZE")) {
      auto const& e_node = train_event.node();
      auto event_type = parse_type(e_node.attribute("Typ").value());
      if (event_type == boost::none) {
        continue;
      }

      auto station_id = parse_station(ctx.b_, e_node);
      auto schedule_time =
          parse_schedule_time(ctx, child_attr(e_node, "Zeit", "Soll").value());

      auto event = CreateEvent(ctx.b_, station_id, service_num, line_id_offset,
                               *event_type, schedule_time);
      func(event, e_node, t_node);
    }
  }
}

Offset<IdEvent> inline parse_trip_id(
    context& ctx, xml_node const& msg,
    char const* service_selector = "./Service") {
  auto const& node = msg.select_node(service_selector).node();

  auto station_id = parse_station(ctx.b_, node, "IdBfEvaNr");
  auto service_num = node.attribute("IdZNr").as_uint();
  auto schedule_time =
      parse_schedule_time(ctx, node.attribute("IdZeit").value());

  std::string reg_sta(node.attribute("RegSta").value());
  auto trip_type = (reg_sta.empty() || reg_sta == "Plan")
                       ? IdEventType_Schedule
                       : IdEventType_Additional;
  // desired side-effect: update temporal bounds
  parse_schedule_time(ctx, node.attribute("Zielzeit").value());

  return CreateIdEvent(ctx.b_, station_id, service_num, schedule_time,
                       trip_type);
}

Offset<FreeText> parse_free_text(context& ctx, xml_node const& msg,
                                 char const* selector = "./FT") {
  auto const& node = msg.select_node(selector).node();
  Range r(0, 0);
  return CreateFreeText(ctx.b_, &r, node.attribute("Code").as_int(),
                        ctx.b_.CreateString(node.attribute("Text").value()),
                        ctx.b_.CreateString(node.attribute("Typ").value()));
}

Offset<Message> parse_delay_msg(context& ctx, xml_node const& msg,
                                DelayType type) {
  std::vector<Offset<UpdatedEvent>> events;
  foreach_event(
      ctx, msg,
      [&](Offset<Event> const& event, xml_node const& e_node, xml_node const&) {
        auto attr_name = (type == DelayType_Is) ? "Ist" : "Prog";
        auto updated =
            parse_time(child_attr(e_node, "Zeit", attr_name).value());
        events.push_back(CreateUpdatedEvent(ctx.b_, event, updated));
      });
  auto trip_id = parse_trip_id(ctx, msg);
  return CreateMessage(
      ctx.b_, ctx.earliest_, ctx.latest_, ctx.timestamp_,
      MessageUnion_DelayMessage,
      CreateDelayMessage(ctx.b_, trip_id, type, ctx.b_.CreateVector(events))
          .Union());
}

Offset<Message> parse_cancel_msg(context& ctx, xml_node const& msg) {
  std::vector<Offset<Event>> events;
  foreach_event(ctx, msg,
                [&](Offset<Event> const& event, xml_node const&,
                    xml_node const&) { events.push_back(event); });
  auto trip_id = parse_trip_id(ctx, msg);
  return CreateMessage(
      ctx.b_, ctx.earliest_, ctx.latest_, ctx.timestamp_,
      MessageUnion_CancelMessage,
      CreateCancelMessage(ctx.b_, trip_id, ctx.b_.CreateVector(events))
          .Union());
}

Offset<Message> parse_track_msg(context& ctx, xml_node const& msg) {
  std::vector<Offset<UpdatedTrack>> events;
  foreach_event(
      ctx, msg,
      [&](Offset<Event> const& event, xml_node const& e_node, xml_node const&) {
        auto updated_track = child_attr(e_node, "Gleis", "Prog").value();
        events.push_back(CreateUpdatedTrack(
            ctx.b_, event, ctx.b_.CreateString(updated_track)));
      });
  auto trip_id = parse_trip_id(ctx, msg);
  return CreateMessage(
      ctx.b_, ctx.earliest_, ctx.latest_, ctx.timestamp_,
      MessageUnion_TrackMessage,
      CreateTrackMessage(ctx.b_, trip_id, ctx.b_.CreateVector(events)).Union());
}

Offset<Message> parse_free_text_msg(context& ctx, xml_node const& msg) {
  std::vector<Offset<Event>> events;
  foreach_event(
      ctx, msg,
      [&](Offset<Event> const& event, xml_node const&, xml_node const&) {
        events.push_back(event);
      },
      "./ListService/Service/ListZug/Zug");
  auto trip_id = parse_trip_id(ctx, msg, "./ListService/Service");
  auto free_text = parse_free_text(ctx, msg);
  return CreateMessage(
      ctx.b_, ctx.earliest_, ctx.latest_, ctx.timestamp_,
      MessageUnion_FreeTextMessage,
      CreateFreeTextMessage(ctx.b_, trip_id, ctx.b_.CreateVector(events),
                            free_text)
          .Union());
}

Offset<Message> parse_addition_msg(context& ctx, xml_node const& msg) {
  std::vector<Offset<AdditionalEvent>> events;
  foreach_event(
      ctx, msg,
      [&](Offset<Event> const& event, xml_node const& e_node,
          xml_node const& t_node) {
        events.push_back(parse_additional_event(ctx.b_, event, e_node, t_node));
      });
  auto trip_id = parse_trip_id(ctx, msg);
  return CreateMessage(
      ctx.b_, ctx.earliest_, ctx.latest_, ctx.timestamp_,
      MessageUnion_AdditionMessage,
      CreateAdditionMessage(ctx.b_, trip_id, ctx.b_.CreateVector(events))
          .Union());
}

Offset<Message> parse_reroute_msg(context& ctx, xml_node const& msg) {
  std::vector<Offset<Event>> cancelled_events;
  foreach_event(ctx, msg,
                [&](Offset<Event> const& event, xml_node const&,
                    xml_node const&) { cancelled_events.push_back(event); });

  std::vector<Offset<ReroutedEvent>> new_events;
  foreach_event(
      ctx, msg,
      [&](Offset<Event> const& event, xml_node const& e_node,
          xml_node const& t_node) {
        auto additional = parse_additional_event(ctx.b_, event, e_node, t_node);
        cstr status_str = e_node.attribute("RegSta").value();
        auto status = (status_str == "Normal") ? RerouteStatus_Normal
                                               : RerouteStatus_UmlNeu;
        new_events.push_back(CreateReroutedEvent(ctx.b_, additional, status));
      },
      "./Service/ListUml/Uml/ListZug/Zug");

  auto trip_id = parse_trip_id(ctx, msg);
  return CreateMessage(
      ctx.b_, ctx.earliest_, ctx.latest_, ctx.timestamp_,
      MessageUnion_RerouteMessage,
      CreateRerouteMessage(ctx.b_, trip_id,
                           ctx.b_.CreateVector(cancelled_events),
                           ctx.b_.CreateVector(new_events))
          .Union());
}

Offset<Message> parse_conn_decision_msg(context& ctx, xml_node const& msg) {
  auto const& from_e_node = msg.child("ZE");
  auto from = parse_standalone_event(ctx, from_e_node);
  if (from == boost::none) {
    throw std::runtime_error("bad from event in RIS conn decision");
  }
  auto from_trip_id = parse_trip_id(ctx, from_e_node);

  std::vector<Offset<ConnectionDecision>> decisions;
  for (auto&& connection : from_e_node.select_nodes("./ListAnschl/Anschl")) {
    auto const& connection_node = connection.node();
    auto const& to_e_node = connection_node.child("ZE");
    auto to = parse_standalone_event(ctx, to_e_node);
    if (to == boost::none) {
      continue;
    }
    auto to_trip_id = parse_trip_id(ctx, to_e_node);

    auto hold = cstr(connection_node.attribute("Status").value()) == "Gehalten";
    decisions.push_back(
        CreateConnectionDecision(ctx.b_, to_trip_id, *to, hold));
  }

  if (decisions.empty()) {
    throw std::runtime_error("zero valid to events in RIS conn decision");
  }

  return CreateMessage(
      ctx.b_, ctx.earliest_, ctx.latest_, ctx.timestamp_,
      MessageUnion_ConnectionDecisionMessage,
      CreateConnectionDecisionMessage(ctx.b_, from_trip_id, *from,
                                      ctx.b_.CreateVector(decisions))
          .Union());
}

Offset<Message> parse_conn_assessment_msg(context& ctx, xml_node const& msg) {
  auto const& from_e_node = msg.child("ZE");
  auto from = parse_standalone_event(ctx, from_e_node);
  if (from == boost::none) {
    throw std::runtime_error("bad from event in RIS conn assessment");
  }
  auto from_trip_id = parse_trip_id(ctx, from_e_node);

  std::vector<Offset<ConnectionAssessment>> assessments;
  for (auto&& connection : from_e_node.select_nodes("./ListAnschl/Anschl")) {
    auto const& connection_node = connection.node();
    auto const& to_e_node = connection_node.child("ZE");
    auto to = parse_standalone_event(ctx, to_e_node);
    if (to == boost::none) {
      continue;
    }
    auto to_trip_id = parse_trip_id(ctx, to_e_node);

    auto a = connection_node.attribute("Bewertung").as_int();
    assessments.push_back(
        CreateConnectionAssessment(ctx.b_, to_trip_id, *to, a));
  }

  if (assessments.empty()) {
    throw std::runtime_error("zero valid to events in RIS conn assessment");
  }

  return CreateMessage(
      ctx.b_, ctx.earliest_, ctx.latest_, ctx.timestamp_,
      MessageUnion_ConnectionAssessmentMessage,
      CreateConnectionAssessmentMessage(ctx.b_, from_trip_id, *from,
                                        ctx.b_.CreateVector(assessments))
          .Union());
}

boost::optional<ris_message> parse_message(xml_node const& msg,
                                           unixtime t_out) {
  using parser_func_t =
      std::function<Offset<Message>(context&, xml_node const&)>;
  static std::map<cstr, parser_func_t> map(
      {{"Ist",
        [](auto&& c, auto&& m) { return parse_delay_msg(c, m, DelayType_Is); }},
       {"IstProg",
        [](auto&& c, auto&& m) {
          return parse_delay_msg(c, m, DelayType_Forecast);
        }},
       {"Ausfall", [](auto&& c, auto&& m) { return parse_cancel_msg(c, m); }},
       {"Zusatzzug",
        [](auto&& c, auto&& m) { return parse_addition_msg(c, m); }},
       {"Umleitung",
        [](auto&& c, auto&& m) { return parse_reroute_msg(c, m); }},
       {"Gleisaenderung",
        [](auto&& c, auto&& m) { return parse_track_msg(c, m); }},
       {"Freitext",
        [](auto&& c, auto&& m) { return parse_free_text_msg(c, m); }},
       /*{"Anschluss",
        [](auto&& c, auto&& m) { return parse_conn_decision_msg(c, m); }},
       {"Anschlussbewertung",
        [](auto&& c, auto&& m) { return parse_conn_assessment_msg(c, m); }}*/});

  auto const& payload = msg.first_child();
  auto it = map.find(payload.name());

  if (it == end(map)) {
    return boost::none;
  }

  context ctx{t_out};
  ctx.b_.Finish(it->second(ctx, payload));
  return {{ctx.earliest_, ctx.latest_, ctx.timestamp_, std::move(ctx.b_)}};
}

void to_ris_message(std::string_view s,
                    std::function<void(ris_message&&)> const& cb,
                    std::string const& tag) {
  utl::verify(tag.empty(), "risml does not support multi-schedule");

  try {
    xml_document d;
    auto r = d.load_buffer(reinterpret_cast<void const*>(s.data()), s.size());
    if (!r) {
      LOG(error) << "bad XML: " << r.description();
      return;
    }

    auto t_out = parse_time(child_attr(d, "Paket", "TOut").value());
    for (auto const& msg : d.select_nodes("/Paket/ListNachricht/Nachricht")) {
      if (auto parsed_message = parse_message(msg.node(), t_out)) {
        cb(std::move(*parsed_message));
      }
    }
  } catch (std::exception const& e) {
    LOG(error) << "unable to parse RIS message: " << e.what();
  } catch (...) {
    LOG(error) << "unable to parse RIS message";
  }
}

std::vector<ris_message> parse(std::string_view s, std::string const& tag) {
  utl::verify(tag.empty(), "risml does not support multi-schedule");
  std::vector<ris_message> msgs;
  to_ris_message(s, [&](ris_message&& m) { msgs.emplace_back(std::move(m)); });
  return msgs;
}

}  // namespace motis::ris::risml
