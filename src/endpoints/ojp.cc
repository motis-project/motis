#include "motis/endpoints/ojp.h"

#include "pugixml.hpp"

#include <unordered_set>

#include "fmt/format.h"

#include "date/date.h"

#include "geo/polyline_format.h"

#include "net/bad_request_exception.h"

#include "nigiri/timetable.h"

#include "motis/adr_extend_tt.h"
#include "motis/tag_lookup.h"
#include "motis/timetable/clasz_to_mode.h"

namespace n = nigiri;

namespace motis::ep {

static auto response_id = std::atomic_size_t{0U};

struct transport_mode {
  std::string_view transport_mode_;
  std::string_view transport_submode_type_;
  std::string_view transport_submode_name_;
};

transport_mode get_transport_mode(std::int64_t const route_type) {
  switch (route_type) {
    case 0: return {"tram", "TramSubmode", ""};
    case 1: return {"metro", "MetroSubmode", ""};
    case 2: return {"rail", "RailSubmode", ""};
    case 3: return {"bus", "BusSubmode", ""};
    case 4: return {"ferry", "", ""};
    case 5: return {"cableway", "TelecabinSubmode", "cableCar"};
    case 6: return {"telecabin", "TelecabinSubmode", "telecabin"};
    case 7: return {"funicular", "", ""};
    case 11: return {"trolleyBus", "", ""};
    case 12: return {"rail", "RailSubmode", "monorail"};

    case 100: return {"rail", "RailSubmode", ""};
    case 101: return {"rail", "RailSubmode", "highSpeedRail"};
    case 102: return {"rail", "RailSubmode", "longDistance"};
    case 103: return {"rail", "RailSubmode", "interregionalRail"};
    case 104: return {"rail", "RailSubmode", "carTransportRailService"};
    case 105: return {"rail", "RailSubmode", "nightRail"};
    case 106: return {"rail", "RailSubmode", "regionalRail"};
    case 107: return {"rail", "RailSubmode", "touristRailway"};
    case 108: return {"rail", "RailSubmode", "railShuttle"};
    case 109: return {"rail", "RailSubmode", "suburbanRailway"};
    case 110: return {"rail", "RailSubmode", "replacementRailService"};
    case 111: return {"rail", "RailSubmode", "specialTrain"};

    case 200: return {"coach", "CoachSubmode", ""};
    case 201: return {"coach", "CoachSubmode", "internationalCoach"};
    case 202: return {"coach", "CoachSubmode", "nationalCoach"};
    case 203: return {"coach", "CoachSubmode", "shuttleCoach"};
    case 204: return {"coach", "CoachSubmode", "regionalCoach"};
    case 205: return {"coach", "CoachSubmode", "specialCoach"};
    case 206: return {"coach", "CoachSubmode", "sightseeingCoach"};
    case 207: return {"coach", "CoachSubmode", "touristCoach"};
    case 208: return {"coach", "CoachSubmode", "commuterCoach"};

    case 400: return {"metro", "MetroSubmode", "urbanRailway"};
    case 401: return {"metro", "MetroSubmode", "metro"};
    case 402: return {"metro", "MetroSubmode", "tube"};

    case 700: return {"bus", "BusSubmode", ""};
    case 701: return {"bus", "BusSubmode", "regionalBus"};
    case 702: return {"bus", "BusSubmode", "expressBus"};
    case 704: return {"bus", "BusSubmode", "localBus"};
    case 709: return {"bus", "BusSubmode", "mobilityBusForRegisteredDisabled"};
    case 710: return {"bus", "BusSubmode", "sightseeingBus"};
    case 711: return {"bus", "BusSubmode", "shuttleBus"};
    case 712: return {"bus", "BusSubmode", "schoolBus"};
    case 713: return {"bus", "BusSubmode", "schoolAndPublicServiceBus"};
    case 714: return {"bus", "BusSubmode", "railReplacementBus"};
    case 715: return {"bus", "BusSubmode", "demandAndResponseBus"};

    case 900: return {"tram", "TramSubmode", ""};
    case 901: return {"tram", "TramSubmode", "cityTram"};
    case 902: return {"tram", "TramSubmode", "localTram"};
    case 903: return {"tram", "TramSubmode", "regionalTram"};
    case 904: return {"tram", "TramSubmode", "sightseeingTram"};
    case 905: return {"tram", "TramSubmode", "shuttleTram"};

    case 1000: return {"water", "", ""};
    case 1100: return {"air", "", ""};

    case 1300: return {"telecabin", "TelecabinSubmode", ""};
    case 1301: return {"telecabin", "TelecabinSubmode", "telecabin"};
    case 1302: return {"telecabin", "TelecabinSubmode", "cableCar"};
    case 1303: return {"lift", "", ""};
    case 1304: return {"telecabin", "TelecabinSubmode", "chairLift"};
    case 1305: return {"telecabin", "TelecabinSubmode", "dragLift"};
    case 1307: return {"telecabin", "TelecabinSubmode", "lift"};

    case 1400: return {"funicular", "FunicularSubmode", "undefinedFunicular"};

    case 1500: return {"taxi", "TaxiSubmode", ""};
    case 1501: return {"taxi", "TaxiSubmode", "communalTaxi"};
    case 1502: return {"taxi", "TaxiSubmode", "waterTaxi"};
    case 1503: return {"taxi", "TaxiSubmode", "railTaxi"};
    case 1504: return {"taxi", "TaxiSubmode", "bikeTaxi"};
    case 1507: return {"taxi", "TaxiSubmode", "allTaxiServices"};

    case 1700:
    default: return {"selfDrive", "", ""};
  }
}

transport_mode to_pt_mode(api::ModeEnum mode) {
  using api::ModeEnum;
  switch (mode) {
    case ModeEnum::AIRPLANE: return {"air", "", ""};
    case ModeEnum::HIGHSPEED_RAIL:
      return {"rail", "RailSubmode", "highSpeedRail"};
    case ModeEnum::LONG_DISTANCE:
      return {"rail", "RailSubmode", "longDistance"};
    case ModeEnum::COACH: return {"coach", "", ""};
    case ModeEnum::NIGHT_RAIL: return {"rail", "RailSubmode", "nightRail"};
    case ModeEnum::REGIONAL_FAST_RAIL:
    case ModeEnum::REGIONAL_RAIL:
      return {"rail", "RailSubmode", "regionalRail"};
    case ModeEnum::SUBURBAN: return {"rail", "RailSubmode", "suburbanRailway"};
    case ModeEnum::SUBWAY: return {"metro", "MetroSubmode", "tube"};
    case ModeEnum::TRAM: return {"tram", "", ""};
    case ModeEnum::BUS: return {"bus", "", ""};
    case ModeEnum::FERRY: return {"water", "", ""};
    case ModeEnum::ODM: return {"bus", "BusSubmode", "demandAndResponseBus"};
    case ModeEnum::FUNICULAR: return {"funicular", "", ""};
    case ModeEnum::AERIAL_LIFT: return {"telecabin", "", ""};
    case ModeEnum::OTHER:
    default: return {"", "", ""};
  }
}

pugi::xml_node append_mode(pugi::xml_node service, transport_mode const m) {
  auto const [transport_mode, submode_type, submode] = m;
  auto mode = service.append_child("Mode");
  mode.append_child("PtMode").text().set(transport_mode);
  if (!submode_type.empty() && !submode.empty()) {
    mode.append_child(fmt::format("siri:{}", submode_type)).text().set(submode);
  }
  return mode;
}

std::string to_upper_ascii(std::string_view input) {
  auto out = std::string{input};
  std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
    return static_cast<char>(std::toupper(c));
  });
  return out;
}

std::string now_timestamp() {
  auto const now = std::chrono::system_clock::now();
  return date::format("%FT%TZ", date::floor<std::chrono::milliseconds>(now));
}

std::string format_coord_param(double const lat, double const lon) {
  auto out = std::ostringstream{};
  out.setf(std::ios::fixed);
  out << std::setprecision(6) << lat << "," << lon;
  return out.str();
}

std::string time_to_iso(openapi::date_time_t const& t) {
  auto out = std::ostringstream{};
  out << t;
  return out.str();
}

std::string duration_to_iso(std::chrono::seconds const dur) {
  auto s = dur.count();
  auto const h = s / 3600;
  s -= h * 3600;
  auto const m = s / 60;
  s -= m * 60;
  return fmt::format("PT{}{}{}",  //
                     h ? fmt::format("{}H", h) : "",
                     m ? fmt::format("{}M", m) : "",
                     s ? fmt::format("{}S", s) : "");
}

void append_text(std::string_view lang,
                 pugi::xml_node node,
                 std::string_view name,
                 std::string_view value) {
  auto text = node.append_child(name).append_child("Text");
  text.append_attribute("xml:lang").set_value(lang);
  text.text().set(value.data());
}

void append_text(pugi::xml_node node, char const* name, auto&& value) {
  node.append_child(name).text().set(value);
}

void append_text(pugi::xml_node node,
                 char const* name,
                 std::optional<std::string> value) {
  if (value.has_value()) {
    node.append_child(name).text().set(*value);
  }
}

void append_position(pugi::xml_node node,
                     geo::latlng const& pos,
                     std::string_view name = "Position") {
  auto geo = node.append_child(name);
  geo.append_child("siri:Latitude").text().set(pos.lat_, 6);
  geo.append_child("siri:Longitude").text().set(pos.lng_, 6);
}

std::string xml_to_str(pugi::xml_document const& doc) {
  auto out = std::ostringstream{};
  doc.save(out, "  ", pugi::format_indent);
  auto result = out.str();
  if (!result.empty() && result.back() == '\n') {
    result.pop_back();
  }
  return result;
}

std::pair<pugi::xml_document, pugi::xml_node> create_ojp_response() {
  auto doc = pugi::xml_document{};

  auto decl = doc.append_child(pugi::node_declaration);
  decl.append_attribute("version").set_value("1.0");
  decl.append_attribute("encoding").set_value("utf-8");

  auto ojp = doc.append_child("OJP");
  ojp.append_attribute("xmlns:siri").set_value("http://www.siri.org.uk/siri");
  ojp.append_attribute("xmlns").set_value("http://www.vdv.de/ojp");
  ojp.append_attribute("version").set_value("2.0");

  auto response = ojp.append_child("OJPResponse");
  auto service_delivery = response.append_child("siri:ServiceDelivery");
  service_delivery.append_child("siri:ResponseTimestamp")
      .text()
      .set(now_timestamp());
  service_delivery.append_child("siri:ProducerRef").text().set("MOTIS");
  service_delivery.append_child("siri:ResponseMessageIdentifier")
      .text()
      .set(++response_id);

  return {std::move(doc), service_delivery};
}

pugi::xml_document build_geocode_response(
    std::string_view language, api::geocode_response const& matches) {
  auto [doc, service_delivery] = create_ojp_response();

  auto location_information =
      service_delivery.append_child("OJPLocationInformationDelivery");
  location_information.append_child("siri:ResponseTimestamp")
      .text()
      .set(now_timestamp());
  location_information.append_child("siri:DefaultLanguage")
      .text()
      .set(language);

  for (auto const& match : matches) {
    auto const stop_ref = match.id_;
    auto const name = match.name_;
    auto place_result = location_information.append_child("PlaceResult");
    auto place = place_result.append_child("Place");
    auto stop_place = place.append_child("StopPlace");
    stop_place.append_child("StopPlaceRef").text().set(stop_ref);

    append_text(language, stop_place, "StopPlaceName", name);
    append_text(language, place, "Name", name);
    append_position(place, {match.lat_, match.lon_}, "GeoPosition");

    if (match.modes_.has_value() && !match.modes_->empty()) {
      append_mode(place, to_pt_mode(match.modes_->front()));
    }

    place_result.append_child("Complete").text().set(true);
    place_result.append_child("Probability").text().set(1);
  }

  return std::move(doc);
}

pugi::xml_document build_map_stops_response(
    std::string_view timestamp,
    std::string_view language,
    std::vector<api::Place> const& stops) {
  auto [doc, service_delivery] = create_ojp_response();

  auto loc_delivery =
      service_delivery.append_child("OJPLocationInformationDelivery");
  loc_delivery.append_child("siri:ResponseTimestamp")
      .text()
      .set(timestamp.data());
  loc_delivery.append_child("siri:DefaultLanguage").text().set(language);

  for (auto const& stop : stops) {
    auto place_result = loc_delivery.append_child("PlaceResult");
    auto place = place_result.append_child("Place");

    if (stop.stopId_.has_value()) {
      auto stop_place = place.append_child("StopPlace");
      stop_place.append_child("StopPlaceRef").text().set(*stop.stopId_);

      auto stop_place_name = stop_place.append_child("StopPlaceName");
      auto stop_place_text = stop_place_name.append_child("Text");
      stop_place_text.append_attribute("xml:lang").set_value(language);
      stop_place_text.text().set(stop.name_);
    }

    auto place_text = place.append_child("Name").append_child("Text");
    place_text.append_attribute("xml:lang").set_value(language);
    place_text.text().set(stop.name_);

    append_position(place, {stop.lat_, stop.lon_}, "GeoPosition");

    if (stop.modes_.has_value() && !stop.modes_->empty()) {
      append_mode(place, to_pt_mode(stop.modes_->front()));
    }

    place_result.append_child("Complete").text().set(true);
    place_result.append_child("Probability").text().set(1);
  }

  return std::move(doc);
}

void add_place(auto const& t,
               std::string_view language,
               n::lang_t const& lang,
               pugi::xml_node places_node,
               api::Place const& p) {
  if (p.parentId_.has_value() && p.parentId_ != p.stopId_) {
    add_place(
        t, language, lang, places_node,
        to_place(&t.tt_, &t.tags_, t.w_, t.pl_, t.matches_, t.ae_, t.tz_, lang,
                 tt_location{t.tags_.get_location(t.tt_, *p.parentId_)}));
  }

  auto place = places_node.append_child("Place");
  append_text(place, "Name", p.name_);
  append_position(place, {p.lat_, p.lon_}, "GeoPosition");

  if (p.parentId_.has_value()) {
    auto sp = place.append_child("StopPoint");
    sp.append_child("siri:StopPointRef").text().set(*p.stopId_);
    append_text(language, sp, "StopPointName", p.name_);
    if (p.parentId_.has_value()) {
      sp.append_child("ParentRef").text().set(*p.parentId_);
    }
  } else {
    auto sp = place.append_child("StopPlace");
    sp.append_child("siri:StopPlaceRef").text().set(*p.stopId_);
    append_text(sp, "StopPlaceName", p.name_);
    if (p.parentId_.has_value()) {
      sp.append_child("ParentRef").text().set(*p.parentId_);
    }
  }
}

pugi::xml_document build_trip_info_response(trip const& trip_ep,
                                            std::string_view language,
                                            std::string_view operating_day,
                                            std::string_view journey_ref,
                                            api::Itinerary const& itinerary,
                                            bool const include_calls,
                                            bool const include_service,
                                            bool const include_track,
                                            bool const include_places,
                                            bool const include_situations) {
  auto [doc, service_delivery] = create_ojp_response();

  auto delivery = service_delivery.append_child("OJPTripInfoDelivery");
  delivery.append_child("siri:ResponseTimestamp").text().set(now_timestamp());
  delivery.append_child("siri:DefaultLanguage").text().set(language.data());

  auto const& leg = itinerary.legs_.at(0);
  auto const lang = n::lang_t{{std::string{language}}};

  if (include_places || include_situations) {
    auto ctx = delivery.append_child("TripInfoResponseContext");

    if (include_places) {
      auto places_node = ctx.append_child("Places");

      for (auto const& stop : {leg.from_, leg.to_}) {
        if (stop.stopId_.has_value()) {
          add_place(trip_ep, language, lang, places_node, stop);
        }
      }

      if (include_calls) {
        for (auto const& stop : leg.intermediateStops_.value()) {
          add_place(trip_ep, language, lang, places_node, stop);
        }
      }
    }

    if (include_situations) {
      ctx.append_child("Situations");
    }
  }

  auto result = delivery.append_child("TripInfoResult");

  if (include_calls) {
    auto add_call = [&, n = 0](api::Place const& place) mutable {
      auto c = result.append_child("PreviousCall");
      c.append_child("siri:StopPointRef").text().set(*place.stopId_);
      append_text(c, "StopPointName", place.name_);
      if (place.scheduledTrack_.has_value()) {
        append_text(c, "PlannedQuay", place.scheduledTrack_);
      }

      auto arr = c.append_child("ServiceArrival");
      append_text(arr, "TimetabledTime",
                  place.scheduledArrival_.transform(time_to_iso));
      append_text(arr, "EstimatedTime", place.arrival_.transform(time_to_iso));

      auto dep = c.append_child("ServiceDeparture");
      append_text(dep, "TimetabledTime",
                  place.scheduledDeparture_.transform(time_to_iso));
      append_text(dep, "EstimatedTime",
                  place.departure_.transform(time_to_iso));

      c.append_child("Order").text().set(++n);
    };

    add_call(leg.from_);
    for (auto const& stop : leg.intermediateStops_.value()) {
      add_call(stop);
    }
    add_call(leg.to_);
  }

  if (include_service) {
    auto service = result.append_child("Service");
    append_text(service, "OperatingDayRef", operating_day);
    append_text(service, "JourneyRef", journey_ref);

    auto const public_code = leg.routeShortName_.value_or(
        leg.displayName_.value_or(leg.routeLongName_.value_or("")));
    if (!public_code.empty()) {
      append_text(service, "PublicCode", public_code);
    }

    append_text(service, "PublicCode", leg.routeShortName_);
    service.append_child("siri:LineRef").text().set(leg.routeId_.value());
    service.append_child("siri:DirectionRef")
        .text()
        .set(leg.directionId_.value());

    auto mode =
        append_mode(service, get_transport_mode(leg.routeType_.value()));
    append_text(language, mode, "Name", leg.displayName_.value());  // TODO Zug?
    append_text(language, mode, "ShortName", leg.displayName_.value());

    append_text(language, service, "PublishedServiceName",
                leg.displayName_.value());
    service.append_child("TrainNumber").text().set(leg.tripShortName_.value());
    append_text(language, service, "OriginText", "n/a");  // TODO
    service.append_child("siri:OperatorRef").text().set(leg.agencyId_.value());
    if (leg.headsign_) {
      append_text(language, service, "DestinationText", *leg.headsign_);
    }
  }

  if (include_track) {
    auto section =
        result.append_child("JourneyTrack").append_child("TrackSection");

    auto start = section.append_child("TrackSectionStart");
    start.append_child("siri:StopPointRef")
        .text()
        .set(leg.from_.stopId_.value());
    append_text(start, "Name", leg.from_.name_);

    auto end = section.append_child("TrackSectionEnd");
    end.append_child("siri:StopPointRef").text().set(leg.to_.stopId_.value());
    append_text(end, "Name", leg.to_.name_);

    auto link = section.append_child("LinkProjection");
    for (auto const& pos : geo::decode_polyline(leg.legGeometry_.points_)) {
      append_position(link, pos);
    }

    append_text(section, "Duration",
                duration_to_iso(std::chrono::seconds{leg.duration_}));
    append_text(section, "Length",
                fmt::format("{}", leg.distance_.value_or(0.0)));
  }

  return std::move(doc);
}

pugi::xml_document build_stop_event_response(
    stop_times const& stop_times_ep,
    std::string_view language,
    bool const include_previous_calls,
    bool const include_onward_calls,
    bool const include_situations,
    api::stoptimes_response const& stop_times_res) {
  auto [doc, service_delivery] = create_ojp_response();

  auto delivery = service_delivery.append_child("OJPStopEventDelivery");
  delivery.append_child("siri:ResponseTimestamp").text().set(now_timestamp());
  delivery.append_child("siri:DefaultLanguage").text().set(language.data());

  auto const lang = n::lang_t{{std::string{language}}};

  {
    auto ctx = delivery.append_child("StopEventResponseContext");
    auto places_node = ctx.append_child("Places");
    auto added = hash_set<std::string>{};

    auto const add_unique = [&](api::Place const& p) {
      if (!added.emplace(p.stopId_.value()).second) {
        return;
      }
      add_place(stop_times_ep, language, lang, places_node, p);
    };

    for (auto const& st : stop_times_res.stopTimes_) {
      add_unique(st.place_);
      if (include_previous_calls && st.previousStops_.has_value()) {
        for (auto const& p : *st.previousStops_) {
          add_unique(p);
        }
      }
      if (include_onward_calls && st.nextStops_.has_value()) {
        for (auto const& p : *st.nextStops_) {
          add_unique(p);
        }
      }
    }

    if (include_situations) {
      ctx.append_child("Situations");
    }
  }

  auto idx = 0;
  for (auto const& st : stop_times_res.stopTimes_) {
    auto result = delivery.append_child("StopEventResult");
    result.append_child("Id").text().set(++idx);

    auto stop_event = result.append_child("StopEvent");

    auto add_call = [&, order = 0](pugi::xml_node parent,
                                   api::Place const& place) mutable {
      auto call = parent.append_child("CallAtStop");
      call.append_child("siri:StopPointRef").text().set(*place.stopId_);
      append_text(call, "StopPointName", place.name_);
      if (place.scheduledTrack_.has_value()) {
        append_text(call, "PlannedQuay", place.scheduledTrack_);
      }

      if (place.scheduledArrival_ || place.arrival_) {
        auto arr = call.append_child("ServiceArrival");
        append_text(arr, "TimetabledTime",
                    place.scheduledArrival_.transform(time_to_iso));
        append_text(arr, "EstimatedTime",
                    place.arrival_.transform(time_to_iso));
      }

      if (place.scheduledDeparture_ || place.departure_) {
        auto dep = call.append_child("ServiceDeparture");
        append_text(dep, "TimetabledTime",
                    place.scheduledDeparture_.transform(time_to_iso));
        append_text(dep, "EstimatedTime",
                    place.departure_.transform(time_to_iso));
      }

      call.append_child("Order").text().set(++order);
    };

    if (include_previous_calls && st.previousStops_.has_value()) {
      for (auto const& p : *st.previousStops_) {
        add_call(stop_event.append_child("PreviousCall"), p);
      }
    }

    {
      auto this_call = stop_event.append_child("ThisCall");
      add_call(this_call, st.place_);
    }

    if (include_onward_calls && st.nextStops_.has_value()) {
      for (auto const& p : *st.nextStops_) {
        add_call(stop_event.append_child("OnwardCall"), p);
      }
    }

    auto service = stop_event.append_child("Service");
    auto const trip_id = split_trip_id(st.tripId_);
    append_text(service, "OperatingDayRef", trip_id.start_date_);
    append_text(service, "JourneyRef", st.tripId_);

    auto const public_code =
        !st.routeShortName_.empty()
            ? st.routeShortName_
            : (!st.displayName_.empty() ? st.displayName_ : st.routeLongName_);
    if (!public_code.empty()) {
      append_text(service, "PublicCode", public_code);
    }

    append_text(service, "PublicCode", st.routeShortName_);
    append_text(service, "siri:LineRef", st.routeId_);
    append_text(service, "siri:DirectionRef", st.directionId_);

    auto mode = append_mode(
        service,
        st.routeType_.has_value()
            ? get_transport_mode(static_cast<std::uint16_t>(*st.routeType_))
            : to_pt_mode(st.mode_));
    append_text(language, mode, "Name", st.displayName_);
    append_text(language, mode, "ShortName", st.routeShortName_);

    append_text(language, service, "PublishedServiceName", st.displayName_);
    append_text(service, "TrainNumber", st.tripShortName_);
    append_text(language, service, "OriginText",
                st.previousStops_
                    .and_then([](std::vector<api::Place> const& x)
                                  -> std::optional<std::string> {
                      return x.empty() ? std::nullopt
                                       : std::optional{x.front().name_};
                    })
                    .value_or("n/a"));
    append_text(service, "siri:OperatorRef", st.agencyId_);
    if (!st.headsign_.empty()) {
      append_text(language, service, "DestinationText", st.headsign_);
    }
  }

  return std::move(doc);
}

net::reply ojp::operator()(net::route_request const& http_req, bool) const {
  auto xml = pugi::xml_document{};
  xml.load_string(http_req.body().c_str());

  auto const req =
      xml.child("OJP").child("OJPRequest").child("siri:ServiceRequest");
  utl::verify<net::bad_request_exception>(
      req, "no OJPReuqest > siri:ServiceRequest found");

  auto response = pugi::xml_document{};
  if (auto const loc_req = req.child("OJPLocationInformationRequest");
      loc_req) {
    auto const input = loc_req.child("InitialInput");
    auto const name = input.child("Name").text();
    auto const geo = input.child("GeoRestriction");
    utl::verify(static_cast<bool>(name) ^ static_cast<bool>(geo),
                "only Name XOR GeoRestriction implemented");

    auto const context = req.child("siri:ServiceRequestContext");
    auto const language = context.child("siri:Language").text().as_string();
    auto const lang = language ? language : std::string{"en"};

    if (name) {
      utl::verify(geocoding_ep_.has_value(), "geocoding not loaded");

      auto const type = to_upper_ascii(
          loc_req.child("Restrictions").child("Type").text().as_string());

      auto url = boost::urls::url{};
      auto params = url.params();
      params.append({"text", name.as_string()});
      params.append({"language", lang});
      if (!type.empty()) {
        params.append({"type", type});
      }

      response = build_geocode_response(lang, (*geocoding_ep_)(url));
    } else if (geo) {
      utl::verify(stops_ep_.has_value(), "stops not loaded");

      auto const rect = geo.child("Rectangle");
      auto const upper_left = rect.child("UpperLeft");
      auto const lower_right = rect.child("LowerRight");
      utl::verify<net::bad_request_exception>(upper_left && lower_right,
                                              "missing GeoRestriction box");

      auto url = boost::urls::url{};
      auto params = url.params();
      params.append(
          {"min",
           fmt::format("{},{}",
                       lower_right.child("siri:Latitude").text().as_double(),
                       upper_left.child("siri:Longitude").text().as_double())});
      params.append(
          {"max",
           fmt::format(
               "{},{}", upper_left.child("siri:Latitude").text().as_double(),
               lower_right.child("siri:Longitude").text().as_double())});
      params.append({"language", lang});

      response =
          build_map_stops_response(now_timestamp(), lang, (*stops_ep_)(url));
    }
  } else if (auto const trip_info_req = req.child("OJPTripInfoRequest")) {
    utl::verify(trip_ep_.has_value(), "trip not loaded");
    auto const context = req.child("siri:ServiceRequestContext");
    auto const language = context.child("siri:Language").text().as_string();
    auto const lang = language ? language : std::string{"en"};

    auto const journey_ref =
        trip_info_req.child("JourneyRef").text().as_string();
    auto const operating_day =
        trip_info_req.child("OperatingDayRef").text().as_string();

    auto const params = trip_info_req.child("Params");

    auto url = boost::urls::url{};
    auto url_params = url.params();
    url_params.append({"tripId", journey_ref});
    url_params.append({"language", lang});

    response = build_trip_info_response(
        *trip_ep_, lang, operating_day, journey_ref, (*trip_ep_)(url),
        params.child("IncludeCalls").text().as_bool(true),
        params.child("IncludeService").text().as_bool(true),
        params.child("IncludeTrackProjection").text().as_bool(true),
        params.child("IncludePlacesContext").text().as_bool(true),
        params.child("IncludeSituationsContext").text().as_bool(true));
  } else if (auto const plan_req = req.child("OJPTripRequest")) {
    throw net::bad_request_exception{"OJP trip request not implemented"};
  } else if (auto const stop_times_req = req.child("OJPStopEventRequest")) {
    utl::verify(stop_times_ep_.has_value(), "stop times not loaded");

    auto const context = req.child("siri:ServiceRequestContext");
    auto const language = context.child("siri:Language").text().as_string();
    auto const lang = language ? language : std::string{"en"};

    auto const location = stop_times_req.child("Location");
    auto const place_ref = location.child("PlaceRef");
    auto const stop_id =
        place_ref.child("siri:StopPointRef").text().as_string();
    auto const dep_arr_time = location.child("DepArrTime").text().as_string();

    auto const params = stop_times_req.child("Params");
    auto const n = params.child("NumberOfResults").text().as_int(10);
    auto const stop_event_type =
        params.child("StopEventType").text().as_string();
    auto const include_prev =
        params.child("IncludePreviousCalls").text().as_bool(false);
    auto const include_onward =
        params.child("IncludeOnwardCalls").text().as_bool(false);

    auto url = boost::urls::url{};
    auto url_params = url.params();
    url_params.append({"stopId", stop_id});
    url_params.append({"language", lang});
    url_params.append({"n", fmt::format("{}", n)});
    if (dep_arr_time && std::string_view{dep_arr_time}.size() != 0) {
      url_params.append({"time", dep_arr_time});
    }
    if (include_prev || include_onward) {
      url_params.append({"fetchStops", "true"});
    }
    if (std::string_view{stop_event_type} == "arrival") {
      url_params.append({"arriveBy", "true"});
    }

    response =
        build_stop_event_response(*stop_times_ep_, lang, include_prev,
                                  include_onward, true, (*stop_times_ep_)(url));
  } else {
    throw net::bad_request_exception{"unsupported OJP request"};
  }

  auto reply = net::web_server::string_res_t{boost::beast::http::status::ok,
                                             http_req.version()};
  reply.insert(boost::beast::http::field::content_type,
               "text/xml; charset=utf-8");
  net::set_response_body(reply, http_req, xml_to_str(response));
  reply.keep_alive(http_req.keep_alive());
  return reply;
}

}  // namespace motis::ep
