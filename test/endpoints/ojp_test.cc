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

using namespace motis;
using namespace date;

constexpr auto const kGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station,platform_code
DA,DA Hbf,49.87260,8.63085,1,,
DA_3,DA Hbf,49.87355,8.63003,0,DA,3
DA_10,DA Hbf,49.87336,8.62926,0,DA,10
FFM,FFM Hbf,50.10701,8.66341,1,,
FFM_101,FFM Hbf,50.10739,8.66333,0,FFM,101
FFM_10,FFM Hbf,50.10593,8.66118,0,FFM,10
FFM_12,FFM Hbf,50.10658,8.66178,0,FFM,12
de:6412:10:6:1,FFM Hbf U-Bahn,50.107577,8.6638173,0,FFM,U4
LANGEN,Langen,49.99359,8.65677,1,,1
FFM_HAUPT,FFM Hauptwache,50.11403,8.67835,1,,
FFM_HAUPT_U,Hauptwache U1/U2/U3/U8,50.11385,8.67912,0,FFM_HAUPT,
FFM_HAUPT_S,FFM Hauptwache S,50.11404,8.67824,0,FFM_HAUPT,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
S3,DB,S3,,,109
U4,DB,U4,,,402
ICE,DB,ICE,,,101

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
S3,S1,S3,,
U4,S1,U4,,
ICE,S1,ICE,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
S3,01:15:00,01:15:00,FFM_101,1,0,0
S3,01:20:00,01:20:00,FFM_HAUPT_S,2,0,0
U4,01:05:00,01:05:00,de:6412:10:6:1,0,0,0
U4,01:10:00,01:10:00,FFM_HAUPT_U,1,0,0
ICE,00:35:00,00:35:00,DA_10,0,0,0
ICE,00:45:00,00:45:00,FFM_10,1,0,0

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1

# frequencies.txt
trip_id,start_time,end_time,headway_secs
S3,01:15:00,25:15:00,3600
ICE,00:35:00,24:35:00,3600
U4,01:05:00,25:01:00,3600
)";

TEST(motis, ojp_requests) {
  auto ec = std::error_code{};
  std::filesystem::remove_all("test/data", ec);

  auto const c =
      config{.osm_ = {"test/resources/test_case.osm.pbf"},
             .timetable_ =
                 config::timetable{.first_day_ = "2019-05-01",
                                   .num_days_ = 2,
                                   .datasets_ = {{"test", {.path_ = kGTFS}}}},
             .street_routing_ = true,
             .osr_footpath_ = true,
             .geocoding_ = true};
  import(c, "test/data", true);
  auto d = data{"test/data", c};
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
