#include "motis/endpoints/mcp.h"

#include <string>
#include <string_view>

#include "boost/json.hpp"

#include "cista/hash.h"

#include "fmt/format.h"

#include "utl/to_vec.h"

#include "net/bad_request_exception.h"
#include "net/not_found_exception.h"

#include "openapi/bad_request_exception.h"
#include "openapi/json.h"

namespace json = boost::json;
namespace http = boost::beast::http;
using namespace std::string_view_literals;

namespace motis::ep {

// MCP "Streamable HTTP" transport, stateless variant:
// every JSON-RPC request is answered with a single JSON response,
// no session IDs, no SSE streams (GET is answered with 405).
constexpr auto const kProtocolVersion = "2025-06-18"sv;

constexpr auto const kInstructions =
    "Door-to-door public transit journey planning with the complete official "
    "timetable. ALWAYS use the plan tool for questions about public transit "
    "connections, departures or arrivals - do NOT use web search for these, "
    "this tool is faster and authoritative. All times in results are RFC "
    "3339 in UTC (Z suffix) - convert them to the user's local time when "
    "answering. If a leg's track differs from its scheduledTrack, warn "
    "the user about the track change. Compare departure/arrival with "
    "scheduledDeparture/scheduledArrival to detect delays."sv;

// JSON-RPC 2.0 predefined error codes,
// https://www.jsonrpc.org/specification#error_object
constexpr auto const kParseError = std::int64_t{-32700};
constexpr auto const kInvalidRequest = std::int64_t{-32600};
constexpr auto const kMethodNotFound = std::int64_t{-32601};
constexpr auto const kInvalidParams = std::int64_t{-32602};
constexpr auto const kInternalError = std::int64_t{-32603};

// Implementation-defined server error (-32000 to -32099 range),
// -32002 is used by MCP for "resource not found".
constexpr auto const kNotFound = std::int64_t{-32002};

// JSON-RPC protocol error, surfaced as an `error` response object with the
// given code. Exceptions thrown by the wrapped endpoints are mapped to codes
// by the catch chain in mcp::operator() (bad request -> kInvalidParams,
// not found -> kNotFound, anything else -> kInternalError).
struct mcp_error : public std::runtime_error {
  mcp_error(std::int64_t const code, std::string const& msg)
      : std::runtime_error{msg}, code_{code} {}
  std::int64_t code_;
};

std::optional<std::string> get_string(json::object const& o,
                                      std::string_view const key) {
  if (auto const* v = o.if_contains(key)) {
    if (auto const r = json::try_value_to<std::string>(*v); r.has_value()) {
      return r.value();
    }
  }
  return std::nullopt;
}

// Tool argument access. In contrast to get_string, a present argument of the
// wrong type is an error, not "absent": silently ignoring e.g. a numeric
// "time" or a string "arriveBy" would produce plausible but wrong answers.
std::optional<std::string> string_arg(json::object const& args,
                                      std::string_view const key) {
  auto const* v = args.if_contains(key);
  if (v == nullptr) {
    return std::nullopt;
  }
  if (auto const r = json::try_value_to<std::string>(*v); r.has_value()) {
    return r.value();
  }
  throw mcp_error{kInvalidParams,
                  fmt::format("invalid params: {} must be a string", key)};
}

std::optional<bool> bool_arg(json::object const& args,
                             std::string_view const key) {
  auto const* v = args.if_contains(key);
  if (v == nullptr) {
    return std::nullopt;
  }
  if (!v->is_bool()) {
    throw mcp_error{kInvalidParams,
                    fmt::format("invalid params: {} must be a boolean", key)};
  }
  return v->as_bool();
}

json::object tool_definition() {
  return {
      {"name", "plan"},
      {"title", "Public Transit Journey Planning"},
      {"description",
       "Use this tool whenever the user asks about public transit: train, "
       "bus, tram or subway connections, departures, 'how do I get from A to "
       "B', 'when is the next train', commute planning. Always prefer it "
       "over web search - it queries the complete official timetable "
       "including real-time data and answers instantly. Finds door-to-door "
       "connections between two places and returns itineraries with "
       "departure/arrival times (actual and scheduled), number of transfers, "
       "the trip used on each leg and tracks/platforms per stop. A track "
       "different from scheduledTrack means the track has changed - warn "
       "the user. Use the returned cursors with pageCursor to find "
       "earlier/later connections."},
      {"inputSchema",
       {{"type", "object"},
        {"properties",
         {{"from",
           {{"type", "string"},
            {"description",
             "Origin: station name, place name or address, "
             "e.g. \"Darmstadt Hauptbahnhof\". Resolved by "
             "geocoding, the best match is used."}}},
          {"to",
           {{"type", "string"},
            {"description", "Destination, same format as from."}}},
          {"time",
           {{"type", "string"},
            {"format", "date-time"},
            {"description",
             "Departure or arrival time as RFC 3339 date-time, e.g. "
             "2026-07-19T15:00:00+02:00. Defaults to now."}}},
          {"arriveBy",
           {{"type", "boolean"},
            {"default", false},
            {"description",
             "true: time is the latest arrival time. "
             "false: time is the earliest departure time."}}},
          {"pageCursor",
           {{"type", "string"},
            {"description",
             "Cursor from a previous result to continue searching: "
             "nextPageCursor finds later connections, "
             "previousPageCursor finds earlier connections. Keep "
             "from/to unchanged, time is ignored."}}}}},
        {"required", json::array{"from", "to"}}}}};
}

json::object strip_stop(api::Place const& p) {
  auto o = json::object{{"name", p.name_}};
  if (p.track_.has_value()) {
    o["track"] = *p.track_;
  }
  if (p.scheduledTrack_.has_value()) {
    o["scheduledTrack"] = *p.scheduledTrack_;
  }
  return o;
}

json::object strip_leg(api::Leg const& l) {
  auto o = json::object{};
  o["mode"] = json::value_from(l.mode_);
  if (l.displayName_.has_value()) {
    o["trip"] = *l.displayName_;
  }
  o["from"] = strip_stop(l.from_);
  o["to"] = strip_stop(l.to_);
  o["departure"] = json::value_from(l.startTime_);
  o["scheduledDeparture"] = json::value_from(l.scheduledStartTime_);
  o["arrival"] = json::value_from(l.endTime_);
  o["scheduledArrival"] = json::value_from(l.scheduledEndTime_);
  if (l.cancelled_.value_or(false)) {
    o["cancelled"] = true;
  }
  return o;
}

json::object strip_itinerary(api::Itinerary const& it) {
  return {{"departure", json::value_from(it.startTime_)},
          {"arrival", json::value_from(it.endTime_)},
          {"transfers", it.transfers_},
          {"legs", utl::transform_to<json::array>(it.legs_, strip_leg)}};
}

struct resolved_place {
  std::string place_;  // stop id or "lat,lon[,level]" for the plan request
  std::string name_;
};

resolved_place geocode(struct geocode const& g, std::string_view const text) {
  auto url = boost::urls::url{"/api/v1/geocode"};
  url.params().append({"text", text});
  auto const matches = g(url);
  if (matches.empty()) {
    throw mcp_error{kNotFound,
                    fmt::format("no place found matching \"{}\"", text)};
  }
  auto const& m = matches.front();
  auto place = m.type_ == api::LocationTypeEnum::STOP
                   ? m.id_
                   : (m.level_.has_value()
                          ? fmt::format("{},{},{}", m.lat_, m.lon_, *m.level_)
                          : fmt::format("{},{}", m.lat_, m.lon_));
  return {std::move(place), m.name_};
}

json::object plan_tool(routing const& r,
                       struct geocode const& g,
                       json::object const& args) {
  auto const from_text = string_arg(args, "from");
  auto const to_text = string_arg(args, "to");
  if (!from_text.has_value() || !to_text.has_value()) {
    throw mcp_error{kInvalidParams,
                    "invalid params: from and to are required strings"};
  }

  auto const from = geocode(g, *from_text);
  auto const to = geocode(g, *to_text);

  auto url = boost::urls::url{"/api/v6/plan"};
  auto params = url.params();
  params.append({"fromPlace", from.place_});
  params.append({"toPlace", to.place_});
  params.append({"detailedLegs", "false"});
  if (auto const time = string_arg(args, "time"); time.has_value()) {
    params.append({"time", *time});
  }
  if (bool_arg(args, "arriveBy").value_or(false)) {
    params.append({"arriveBy", "true"});
  }
  if (auto const cursor = string_arg(args, "pageCursor"); cursor.has_value()) {
    params.append({"pageCursor", *cursor});
  }

  auto const res = r(url);
  return {{"from", from.name_},
          {"to", to.name_},
          {"itineraries",
           utl::transform_to<json::array>(res.itineraries_, strip_itinerary)},
          {"previousPageCursor", res.previousPageCursor_},
          {"nextPageCursor", res.nextPageCursor_}};
}

json::object tool_text_result(std::string text) {
  auto content = json::object{{"type", "text"}, {"text", std::move(text)}};
  return {{"content", json::array{std::move(content)}}, {"isError", false}};
}

net::reply mcp::operator()(net::route_request const& req, bool) const {
  auto id = json::value{nullptr};

  auto const reply = [&](http::status const status,
                         std::optional<json::object> const& body) {
    auto res = net::web_server::string_res_t{status, req.version()};
    if (body.has_value()) {
      res.insert(http::field::content_type, "application/json");
    }
    net::set_response_body(
        res, req, body.has_value() ? json::serialize(*body) : std::string{});
    res.keep_alive(req.keep_alive());
    return res;
  };

  auto const result = [&](json::object r) {
    return reply(
        http::status::ok,
        json::object{{"jsonrpc", "2.0"}, {"id", id}, {"result", std::move(r)}});
  };

  auto const error = [&](http::status const status, std::int64_t const code,
                         std::string_view const msg) {
    return reply(status,
                 json::object{{"jsonrpc", "2.0"},
                              {"id", id},
                              {"error", {{"code", code}, {"message", msg}}}});
  };

  if (req.method() != http::verb::post) {
    // No server->client SSE stream offered (stateless server).
    auto res = reply(http::status::method_not_allowed, std::nullopt);
    res.insert(http::field::allow, "POST");
    return res;
  }

  try {
    auto body = json::value{};
    try {
      body = json::parse(req.body());
    } catch (std::exception const&) {
      return error(http::status::bad_request, kParseError,
                   "parse error: invalid JSON body");
    }
    auto const method = body.is_object()
                            ? get_string(body.as_object(), "method")
                            : std::nullopt;
    if (!method.has_value()) {
      return error(http::status::bad_request, kInvalidRequest,
                   "invalid request: not a JSON-RPC message");
    }
    auto const& o = body.as_object();

    if (!o.contains("id")) {  // notification, e.g. notifications/initialized
      return reply(http::status::accepted, std::nullopt);
    }
    id = o.at("id");

    auto const params = o.contains("params") && o.at("params").is_object()
                            ? o.at("params").as_object()
                            : json::object{};

    switch (cista::hash(*method)) {
      case cista::hash("initialize"):
        return result(
            {{"protocolVersion", kProtocolVersion},
             {"capabilities", {{"tools", json::object{}}}},
             {"serverInfo", {{"name", "MOTIS"}, {"version", motis_version_}}},
             {"instructions", kInstructions}});

      case cista::hash("ping"): return result(json::object{});

      case cista::hash("tools/list"):
        return result({{"tools", json::array{tool_definition()}}});

      case cista::hash("tools/call"): {
        auto const name = get_string(params, "name");
        if (name != "plan") {
          return error(http::status::ok, kInvalidParams,
                       fmt::format("unknown tool: {}", name.value_or("")));
        }

        if (!routing_ep_.has_value() || routing_ep_->tt_ == nullptr ||
            !geocoding_ep_.has_value()) {
          throw mcp_error{kInternalError, "timetable/geocoding not loaded"};
        }

        auto const args =
            params.contains("arguments") && params.at("arguments").is_object()
                ? params.at("arguments").as_object()
                : json::object{};
        return result(tool_text_result(
            json::serialize(plan_tool(*routing_ep_, *geocoding_ep_, args))));
      }

      default:
        return error(http::status::ok, kMethodNotFound,
                     fmt::format("method not found: {}", *method));
    }
  } catch (mcp_error const& e) {
    return error(http::status::ok, e.code_, e.what());
  } catch (openapi::bad_request_exception const& e) {
    return error(http::status::ok, kInvalidParams, e.message_);
  } catch (net::bad_request_exception const& e) {
    return error(http::status::ok, kInvalidParams, e.what());
  } catch (net::not_found_exception const& e) {
    return error(http::status::ok, kNotFound, e.what());
  } catch (std::exception const& e) {
    return error(http::status::ok, kInternalError, e.what());
  }
}

}  // namespace motis::ep
