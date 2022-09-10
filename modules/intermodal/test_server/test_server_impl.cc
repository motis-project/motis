#include "test_server_impl.h"

#include <iostream>
#include <memory>
#include <ctime>

#include "ctx/call.h"
#include "rapidjson/rapidjson.h"
#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "motis/json/json.h"

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

std::string create_resbody(net::test_server::http_req_t const& req, bool post)
{
  if(post)
  {
    Document document;
    if (document.Parse(req.body().c_str()).HasParseError())
    {
      document.GetParseError();
      throw utl::fail("Test Server create result body: Bad JSON: {} at offset {}",
                      GetParseError_En(document.GetParseError()),
                      document.GetErrorOffset());
    }
    auto const& data = get_obj(document, "data");
    auto read_jay_key_double = [&](char const* key, char const* name) -> double
    {
      auto const it = data.FindMember(key);
      if (it != data.MemberEnd() && it->value.IsDouble())
      {
        return it->value.GetDouble();
      }
      else if(it != data.MemberEnd() && it->value.HasMember(name))
      {
        auto const at = it->value.FindMember(name);
        if(at->value.IsDouble())
        {
          return at->value.GetDouble();
        }
      }
      return -1.0;
    };
    double startlat = read_jay_key_double("origin", "lat");
    double startlng = read_jay_key_double("origin", "lng");
    double endlat = read_jay_key_double("destination", "lat");
    double endlng = read_jay_key_double("destination", "lng");

    time_t timenow = time(nullptr);
    timenow += minutes * 60;
    using time_point = std::chrono::system_clock::time_point;
    time_point time_convertion_dep{std::chrono::duration_cast<time_point::duration>(std::chrono::seconds(timenow))};
    std::string s_time_dep = date::format("%FT%TZ", date::floor<std::chrono::seconds>(time_convertion_dep));
    timenow += 900;
    time_point time_convertion_arr{std::chrono::duration_cast<time_point::duration>(std::chrono::seconds(timenow))};
    std::string s_time_arr = date::format("%FT%TZ", date::floor<std::chrono::seconds>(time_convertion_arr));

    int walk = 0;
    int walk2 = 10;
    auto res = R"( { "data": {
                      "id": "rid_12345-abcde-1a2b3c",
                      "created_at": "2017-09-06T15:08:43Z",
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
            R"( "walking_duration": )" + std::to_string(walk) + "," +
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
            R"( "walking_duration": )" + std::to_string(walk2) + "," +
            R"( "walking_track": "_sdpH_y|u@_pR_pR_pR_pR_pR_pR_pR_pR"},
                "fare": {
                        "type": "fare",
                        "id": "far_12345-abcde-1a2b3c-2fed0f810837",
                        "final_price": 15,
                        "currency": "EUR"}}
            } )";
    return res;
  }
  else
  {
      auto result = R"( {
            "data": {
              "id": "1234567890",
              "area": {
                    "type": "MultiPolygon",
                    "coordinates": [[[
                      [47.36195,7.29655],
                      [47.47063,7.61869],
                      [47.34102,7.92639],
                      [47.17985,7.61044],
                      [47.27586,7.34807],
                      [47.36195,7.29655]
                      ]]]},
              "message": "This is a default message"
    }})";
    return result;
  }
}

struct test_server::impl {
    impl(boost::asio::io_service& ios)
        : ios{ios}, serve{ios} {}

    void listen_tome(std::string const& host, std::string const& port,
                     boost::system::error_code& erco)
    {
      serve.on_http_request([this](net::test_server::http_req_t const& req,
                                   net::test_server::http_res_cb_t const& cb, bool)
                            { on_http_request(req, cb); });
      serve.on_upgrade_ok([](net::test_server::http_req_t const& req) {
        return req.target() == "/" ;
      });
      serve.init(host, port, erco);
      serve.set_timeout(std::chrono::seconds(5*60));
      if (erco) {
        std::cout << "testserver: init error: " << erco.message() << "\n";
      }
      std::cout << "testserver is running on http://" + host + ":" + port + "/ \n "
                  "info: " + erco.message() + "\n";
      serve.run();
    }

    void stop_it() { serve.stop(); std::cout << "testserver: stopped \n";}

    void on_http_request(net::test_server::http_req_t const& req,
                         net::test_server::http_res_cb_t const& cb)
    {
      switch(req.method())
      {
        case verb::options:
        {
          std::string_view resbody = "allow: post, head, get";
          status status = status::ok;
          std::string_view contenttype = "text/html";
          cb(string_response(req, resbody, status, contenttype));
          break;
        }
        case verb::post:
        {
          std::string sres = create_resbody(req, true);
          minutes += 5;
          std::string_view resbody{sres};
          if(req.body().empty())
          {
            cb(server_error_response(req, "SEND REQUEST BODY"));
            break;
          }
          status status = status::ok;
          std::string_view contenttype = "application/json";
          cb(string_response(req, resbody, status, contenttype));
          break;
        }
        case verb::head:
        {
          status status = status::ok;
          std::string_view contenttype = "text/html";
          cb(empty_response(req, status, contenttype));
          break;
        }
        case verb::get: {
          std::string sres = create_resbody(req, false);
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
    boost::asio::io_service& ios;
    net::test_server serve;
};

  test_server::test_server(boost::asio::io_service& ios)
      : impl_(new impl(ios)) {}

  test_server::~test_server() = default;

  void test_server::listen_tome(std::string const& host, std::string const& port,
                                boost::system::error_code& eric) {
    impl_->listen_tome(host, port, eric);
  }

  void test_server::stop_it() { impl_->stop_it(); }

} // namespace motis::intermodal


