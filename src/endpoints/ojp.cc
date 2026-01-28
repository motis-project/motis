#include "motis/endpoints/ojp.h"

#include "pugixml.hpp"

#include "date/date.h"
#include "net/bad_request_exception.h"

#include "nigiri/timetable.h"
#include "motis/adr_extend_tt.h"
#include "motis/tag_lookup.h"
#include "motis/timetable/clasz_to_mode.h"

namespace motis::ep {

static auto response_id = std::atomic_size_t{0U};

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

std::string to_pt_mode(api::ModeEnum mode) {
  switch (mode) {
    case api::ModeEnum::BUS: return "bus";
    case api::ModeEnum::TRAM: return "tram";
    case api::ModeEnum::SUBWAY: return "metro";
    case api::ModeEnum::FERRY: return "water";
    case api::ModeEnum::AIRPLANE: return "air";
    case api::ModeEnum::RAIL:
    case api::ModeEnum::REGIONAL_RAIL:
    case api::ModeEnum::HIGHSPEED_RAIL:
    case api::ModeEnum::COACH: return "rail";
    default: return "other";
  }
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

  auto loc_delivery =
      service_delivery.append_child("OJPLocationInformationDelivery");
  loc_delivery.append_child("siri:ResponseTimestamp")
      .text()
      .set(now_timestamp());
  loc_delivery.append_child("siri:DefaultLanguage").text().set(language);

  for (auto const& match : matches) {
    auto const stop_ref = match.id_;
    auto const name = match.name_;
    auto place_result = loc_delivery.append_child("PlaceResult");
    auto place = place_result.append_child("Place");
    auto stop_place = place.append_child("StopPlace");
    stop_place.append_child("StopPlaceRef").text().set(stop_ref.c_str());

    auto stop_place_name = stop_place.append_child("StopPlaceName");
    auto stop_place_text = stop_place_name.append_child("Text");
    stop_place_text.append_attribute("xml:lang").set_value(language);
    stop_place_text.text().set(name.c_str());

    auto place_text = place.append_child("Name").append_child("Text");
    place_text.append_attribute("xml:lang").set_value(language);
    place_text.text().set(name.c_str());

    auto geo_pos = place.append_child("GeoPosition");
    geo_pos.append_child("siri:Latitude").text().set(match.lat_, 6);
    geo_pos.append_child("siri:Longitude").text().set(match.lon_, 6);

    if (match.modes_.has_value() && !match.modes_->empty()) {
      place.append_child("Mode").append_child("PtMode").text().set(
          to_pt_mode(match.modes_->front()).c_str());
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
      stop_place_text.text().set(stop.name_.c_str());
    }

    auto place_text = place.append_child("Name").append_child("Text");
    place_text.append_attribute("xml:lang").set_value(language);
    place_text.text().set(stop.name_.c_str());

    auto geo_pos = place.append_child("GeoPosition");
    geo_pos.append_child("siri:Latitude").text().set(stop.lat_, 6);
    geo_pos.append_child("siri:Longitude").text().set(stop.lon_, 6);

    if (stop.modes_.has_value() && !stop.modes_->empty()) {
      auto mode = place.append_child("Mode");
      mode.append_child("PtMode").text().set(
          to_pt_mode(stop.modes_->front()).c_str());
    }

    place_result.append_child("Complete").text().set(true);
    place_result.append_child("Probability").text().set(1);
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
    auto const lang = language ? language : std::string{"de"};

    if (name) {
      utl::verify(geocoding_.has_value(), "geocoding not loaded");

      auto const type = to_upper_ascii(
          loc_req.child("Restrictions").child("Type").text().as_string());

      auto url = boost::urls::url{};
      auto params = url.params();
      params.append({"text", name.as_string()});
      params.append({"language", lang});
      if (!type.empty()) {
        params.append({"type", type});
      }

      response = build_geocode_response(lang, (*geocoding_)(url));
    } else if (geo) {
      utl::verify(s_.has_value(), "stops not loaded");

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

      response = build_map_stops_response(now_timestamp(), lang, (*s_)(url));
    }
  } else if (auto const plan_req = req.child("OJPTripRequest")) {
    throw net::bad_request_exception{"OJP trip request not implemented"};
  } else if (auto const stop_times_req = req.child("OJPStopEventRequest")) {
    throw net::bad_request_exception{"OJP stop event request not implemented"};
  } else if (auto const trip_info_req = req.child("OJPTripInfoRequest")) {
    throw net::bad_request_exception{"OJP trip info request not implemented"};
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
