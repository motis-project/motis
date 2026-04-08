#include "gtest/gtest.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

#include "date/date.h"

#include "utl/init_from.h"
#include "utl/read_file.h"

#include "adr/formatter.h"

#include "motis/config.h"
#include "motis/data.h"
#include "motis/endpoints/ojp.h"
#include "motis/import.h"

#include "../test_case.h"

using namespace motis;
using namespace date;

TEST(motis, ojp_requests) {
  auto [d, _] = get_test_case<test_case::FFM_ojp_requests>();
  d.init_rtt(date::sys_days{2019_y / May / 1});

  auto const ojp_ep = ep::ojp{
      .routing_ep_ = utl::init_from<ep::routing>(d),
      .geocoding_ep_ = utl::init_from<ep::geocode>(d),
      .stops_ep_ = utl::init_from<ep::stops>(d),
      .stop_times_ep_ = utl::init_from<ep::stop_times>(d),
      .trip_ep_ = utl::init_from<ep::trip>(d),
  };

  auto const send_request = [&](std::string_view body) {
    net::web_server::http_req_t req{boost::beast::http::verb::post,
                                    "/api/v2/ojp", 11};
    req.set(boost::beast::http::field::content_type, "text/xml; charset=utf-8");
    req.body() = std::string{body};
    req.prepare_payload();
    return ojp_ep(net::route_request{std::move(req)}, false);
  };

  auto const normalize_response = [](std::string_view input) {
    auto out = std::string{input};

    auto const normalize_tag = [&](std::string_view start_tag,
                                   std::string_view end_tag,
                                   std::string_view replacement) {
      auto pos = std::size_t{0};
      while ((pos = out.find(start_tag, pos)) != std::string::npos) {
        auto const value_start = pos + start_tag.size();
        auto const value_end = out.find(end_tag, value_start);
        if (value_end == std::string::npos) {
          break;
        }
        out.replace(value_start, value_end - value_start, replacement);
        pos = value_start + replacement.size() + end_tag.size();
      }
    };

    normalize_tag("<siri:ResponseTimestamp>", "</siri:ResponseTimestamp>",
                  "NOW");
    normalize_tag("<siri:ResponseMessageIdentifier>",
                  "</siri:ResponseMessageIdentifier>", "MSG");
    normalize_tag("<LinkProjection>", "</LinkProjection>", "");

    return out;
  };

  auto const expect_response = [&](char const* request_path,
                                   char const* response_path) {
    auto const request = utl::read_file(request_path).value();
    auto expected = utl::read_file(response_path).value();
    auto const reply = send_request(request);
    auto const* res = std::get_if<net::web_server::string_res_t>(&reply);
    ASSERT_NE(nullptr, res);
    EXPECT_EQ(boost::beast::http::status::ok, res->result());
    EXPECT_EQ("text/xml; charset=utf-8",
              res->base()[boost::beast::http::field::content_type]);
    EXPECT_EQ(normalize_response(expected), normalize_response(res->body()));
  };

  expect_response("test/resources/ojp/geocoding_request.xml",
                  "test/resources/ojp/geocoding_response.xml");
  expect_response("test/resources/ojp/map_stops_request.xml",
                  "test/resources/ojp/map_stops_response.xml");
  expect_response("test/resources/ojp/stop_event_request.xml",
                  "test/resources/ojp/stop_event_response.xml");
  expect_response("test/resources/ojp/trip_info_request.xml",
                  "test/resources/ojp/trip_info_response.xml");
  expect_response("test/resources/ojp/routing_request.xml",
                  "test/resources/ojp/routing_response.xml");
  expect_response("test/resources/ojp/intermodal_routing_request.xml",
                  "test/resources/ojp/intermodal_routing_response.xml");
}
