#include "test_server_impl.h"

#include <iostream>
#include <memory>
#include <ctime>
#include <utility>
#include <thread>

#include "rapidjson/rapidjson.h"
#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "motis/json/json.h"

#include "ctx/call.h"
#include "motis/module/receiver.h"
#include "test_server.h"
#include "net/web_server/responses.h"
#include "date/date.h"

namespace motis::intermodal {

using namespace net;
using namespace boost::beast::http;
using namespace motis::json;
using namespace rapidjson;

int minutes = 0;
int count = 0;

std::string create_resbody(net::test_server::http_req_t const& req, bool post, int area) {
  if(post) {
    Document document;
    if (document.Parse(req.body().c_str()).HasParseError()) {
      document.GetParseError();
      throw utl::fail("Test Server create result body: Bad JSON: {} at offset {}",
                      GetParseError_En(document.GetParseError()),
                      document.GetErrorOffset());
    }
    auto const& data = get_obj(document, "data");
    auto read_json_key_double = [&](char const* key, char const* name) -> double {
      auto const it = data.FindMember(key);
      if (it != data.MemberEnd() && it->value.IsDouble()) {
        return it->value.GetDouble();
      }
      else if(it != data.MemberEnd() && it->value.HasMember(name)) {
        auto const at = it->value.FindMember(name);
        if(at->value.IsDouble()) {
          return at->value.GetDouble();
        }
      }
      return -1.0;
    };
    auto read_json_key_string = [&](char const* key, char const* name) -> std::string {
      auto const it = data.FindMember(key);
      if (it != data.MemberEnd() && it->value.IsString()) {
        return it->value.GetString();
      }
      else if(it != data.MemberEnd() && it->value.HasMember(name)) {
        auto const at = it->value.FindMember(name);
        if(at->value.IsString()) {
          return at->value.GetString();
        }
      }
      return "";
    };
    double startlat = read_json_key_double("origin", "lat");
    double startlng = read_json_key_double("origin", "lng");
    double endlat = read_json_key_double("destination", "lat");
    double endlng = read_json_key_double("destination", "lng");
    if(startlat == -1.0 || startlng == -1.0 || endlat == -1.0 || endlng == -1.0) {
      return "";
    }
    std::string departure = read_json_key_string("origin", "time");
    std::string arrival = read_json_key_string("destination", "time");
    if(departure.empty() || arrival.empty()) {
      return "";
    }

    auto traveltime_to_unixtime = [&](std::string const& timestring) -> date::sys_seconds {
      std::istringstream in(timestring);
      date::sys_seconds tp;
      in >> date::parse("%FT%TZ", tp);
      if (in.fail()) {
        in.clear();
        in.str(timestring);
        in >> date::parse("%FT%T%z", tp);
      }
      return tp;
    };

    time_t tests_time_dep = traveltime_to_unixtime(departure).time_since_epoch().count();
    time_t tests_time_arr = traveltime_to_unixtime(arrival).time_since_epoch().count();
    if(count%4==0) {
      minutes = 0;
    }
    tests_time_dep += minutes * 60;
    using time_point = std::chrono::system_clock::time_point;
    time_point time_convertion_dep{std::chrono::duration_cast<time_point::duration>(std::chrono::seconds(tests_time_dep))};
    std::string s_time_dep = date::format("%FT%TZ", date::floor<std::chrono::seconds>(time_convertion_dep));
    time_t diff = (tests_time_arr - tests_time_dep) + minutes;
    tests_time_dep += (diff - 120); // wie lange die Fahrt dauert
    time_point time_convertion_arr{std::chrono::duration_cast<time_point::duration>(std::chrono::seconds(tests_time_dep))};
    std::string s_time_arr = date::format("%FT%TZ", date::floor<std::chrono::seconds>(time_convertion_arr));

    int walk_before = 0;
    int walk_after = 0;
    if(count%2==0) {
      walk_before+= 120;
      walk_after = 0;
    }
    if(count%3==0) {
      walk_after += 60;
    }
    if(count%4==0) {
      walk_before = 0;
      walk_after = 0;
    }
    std::string id = "rid_12345-abcde-1a2b3c-" + std::to_string(count);
    auto res = R"( { "data": {
                      "id": ")" + id + "\"," +
               R"( "created_at": "2017-09-06T15:08:43Z",
                      "updated_at": "2017-09-06T15:08:43Z",
                      "type": "ride",
                      "product_id": "prd_12345-abcde-1a2b3c-1a2b3cca33e34",
                  "pickup": {
                            "id": "cap_12345-abcde-1a2b3c-e46dc46c6e39",
                            "type": "calculated_point",
                            "waypoint_type": "pickup",
                            "time": "2022-08-08T15:20:00Z",
                            "negotiation_time": ")" + s_time_dep + "\"," +
              R"( "negotiation_time_max": "2022-08-08T15:20:00Z",
                "lat": )" + std::to_string(startlat) + "," +
            R"( "lng": )" + std::to_string(startlng) + "," +
            R"( "walking_duration": )" + std::to_string(walk_before) + "," +
            R"( "walking_track": "_iajH_oyo@_pR_pR_pR_pR_pR_pR_pR_pR"},
                "dropoff": {
                            "id": "cap_12345-abcde-1a2b3c-d6a51d19b7a0",
                            "type": "calculated_point",
                            "time": "2022-08-06T15:42:00Z",
                            "negotiation_time": ")" + s_time_arr + "\"," +
            R"( "negotiation_time_max": "2022-08-08T15:40:00Z",
                "waypoint_type": "dropoff",
                "lat": )" + std::to_string(endlat) + "," +
            R"( "lng": )" + std::to_string(endlng) + "," +
            R"( "walking_duration": )" + std::to_string(walk_after) + "," +
            R"( "walking_track": "_sdpH_y|u@_pR_pR_pR_pR_pR_pR_pR_pR"},
                "fare": {
                        "type": "fare",
                        "id": "far_12345-abcde-1a2b3c-2fed0f810837",
                        "final_price": 15,
                        "currency": "EUR"}}
            } )";
    return res;
  } else {
    auto result = "";
    // hier wieder leeren string einfuegen, damits auch mal direkt nicht verfuegbar ist
    if(area == 0) {
      result = R"( {
            "data": {
              "id": "1234567890",
              "area": {
                    "type": "MultiPolygon",
                    "coordinates": [[[
                      [47.53852,9.54136],
                      [47.78990,8.57427],
                      [47.56816,7.50827],
                      [47.53852,6.95879],
                      [46.94976,6.43129],
                      [46.47554,6.06863],
                      [45.93611,7.01374],
                      [45.94374,7.85994],
                      [46.46042,8.40942],
                      [46.02765,8.79406],
                      [46.52844,9.32157],
                      [46.22548,10.13480],
                      [46.84473,10.46449],
                      [47.09196,9.46443],
                      [47.53852,9.54136]
                      ]]]},
              "message": "This is a default message"
    }})";
    }
    return result;
  }
}

struct test_server::impl {
    impl(boost::asio::io_service& ios, std::vector<std::string> server_arguments)
        : ios_{ios}, serve_{ios}, server_argv_{std::move(server_arguments)} {}

    void listen_tome(std::string const& host, std::string const& port,
                     boost::system::error_code& erco) const {
      serve_.on_http_request([this](net::test_server::http_req_t const& req,
                                   net::test_server::http_res_cb_t const& cb, bool)
                            { on_http_request(req, cb); });
      serve_.on_upgrade_ok([](net::test_server::http_req_t const& req) {
        return req.target() == "/" ;
      });
      serve_.init(host, port, erco);
      serve_.set_timeout(std::chrono::seconds(5*60));
      serve_.set_request_body_limit(1024 * 1024);
      serve_.set_request_queue_limit(1001);
      if (erco) {
        std::cout << "testserver: init error: " << erco.message() << "\n";
      }
      std::cout << "testserver is running on http://" + host + ":" + port + "/ \n "
                  "info: " + erco.message() + "\n";
      serve_.run();
    }

    void stop_it() const { serve_.stop(); std::cout << "testserver: stopped \n";}

    void on_http_request(net::test_server::http_req_t const& req,
                         net::test_server::http_res_cb_t const& cb) const {
      int area = 0;
      for(auto const& s : server_argv_) {
        if(s == "medium") {
          printf("Angekommen und funktioniert\n");
          std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        else if(s == "high") {
          std::this_thread::sleep_for(std::chrono::seconds(2));
        }
        else if(s == "1") {
          area = 1;
        }
        else if(s == "2") {
          area = 2;
        }
        else if(s == "3") {
          area = 3;
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      switch(req.method()) {
        case verb::options: {
          std::string_view resbody = "allow: post, head, get";
          status status = status::ok;
          std::string_view contenttype = "text/html";
          cb(string_response(req, resbody, status, contenttype));
          break;
        }
        case verb::post: {
          if(req.body().empty()) {
            cb(server_error_response(req, "SEND REQUEST BODY"));
            break;
          }
          std::string sres = create_resbody(req, true, area);
          minutes += 5;
          count++;
          std::string_view resbody{sres};
          status status = status::ok;
          std::string_view contenttype = "application/json";
          cb(string_response(req, resbody, status, contenttype));
          break;
        }
        case verb::head: {
          status status = status::ok;
          std::string_view contenttype = "text/html";
          cb(empty_response(req, status, contenttype));
          break;
        }
        case verb::get: {
          count++;
          std::string sres = create_resbody(req, false, area);
          std::string_view resbody{sres};
          status status = status::ok;
          std::string_view contenttype = "application/json";
          cb(string_response(req, resbody, status, contenttype));
          break;
        }
        default:
          cb(server_error_response(req, "SERVER ERROR!"));
      }
    }
    boost::asio::io_service& ios_;
    net::test_server serve_;
    std::vector<std::string> server_argv_;
};

  test_server::test_server(boost::asio::io_service& ios, std::vector<std::string> server_args)
      : impl_(new impl(ios, std::move(server_args))) {}

  test_server::~test_server() = default;

  void test_server::listen_tome(std::string const& host, std::string const& port,
                                boost::system::error_code& ec) {
    impl_->listen_tome(host, port, ec);
  }

  void test_server::stop_it() { impl_->stop_it(); }

} // namespace motis::intermodal


