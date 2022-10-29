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

int walk_before = 0;
int walk_after = 0;
int area = 0;
int minutes = 0;
int count = 0;
bool on = false;
int sleeping_ms = 100;
std::string last_customer("0");
std::vector<std::vector<int>> fleet;

std::string create_resbody(net::test_server::http_req_t const& req, bool post, int zone) {
  //std::this_thread::sleep_for(std::chrono::milliseconds(sleeping_ms));
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
    std::string customer_id = read_json_key_string("product_id", "");
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
    if(count%3==0) {
      minutes = 0;
    }
    tests_time_dep += minutes * 60;
    tests_time_arr += minutes * 60;
    if(count%4==0) {
      tests_time_arr += 2*minutes * 60;
    }
    using time_point = std::chrono::system_clock::time_point;
    time_point time_convertion_dep{std::chrono::duration_cast<time_point::duration>(std::chrono::seconds(tests_time_dep))};
    std::string s_time_dep = date::format("%FT%TZ", date::floor<std::chrono::seconds>(time_convertion_dep));
    time_t diff = tests_time_arr - tests_time_dep;
    tests_time_dep += (diff - 180); // wie lange die Fahrt dauert
    time_point time_convertion_arr{std::chrono::duration_cast<time_point::duration>(std::chrono::seconds(tests_time_dep))};
    std::string s_time_arr = date::format("%FT%TZ", date::floor<std::chrono::seconds>(time_convertion_arr));

    if(on) {
      if(last_customer != customer_id) {
        last_customer = customer_id;
        size_t indexT_dep = s_time_dep.find('T');
        std::string hour_start = s_time_dep.substr(indexT_dep + 1, 2);
        size_t indexT_arr = s_time_dep.find('T');
        std::string hour_end = s_time_dep.substr(indexT_arr + 1, 2);
        int free_from = std::stoi(hour_start);
        int free_to = std::stoi(hour_end);
        bool no_one_free = true;
        for (auto& k : fleet) {
          if (free_from == free_to) {
            if (k.at(free_from)) {
              k.at(free_from) = 0;
              no_one_free = false;
              break;
            }
          } else {
            int all_free = 0;
            int while_count = 0;
            if(free_from == 23 && free_to == 0) {
              if(k.at(free_from) && k.at(free_to)) {
                k.at(free_from) = 0;
                k.at(free_to) = 0;
                no_one_free = false;
                break;
              }
            }
            while(free_from <= free_to) {
              if(k.at(free_from)) {
                k.at(free_from) = 0;
                all_free++;
              }
              free_from++;
              while_count++;
            }
            if(while_count == all_free && while_count != 0 && all_free != 0) {
              no_one_free = false;
            }
            break;
          }
        }
        if(no_one_free) {
          return "";
        }
      }
    }

    /*for(auto f : fleet)
    {
      for(int i = 0; i < f.size(); i++)
      {
        printf("%d: %d \t", i, f.at(i));
      }
      printf("\n");
    }
    printf("\n");*/

    walk_before += 60;
    walk_after += 60;
    if(count%3==0) {
      walk_before = 0;
    }
    if(count%4==0) {
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
    if(zone == 0) {
      // Swiss complete
      result = R"( {
            "data": {
              "id": "swiss complete",
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
    else if(zone == 1) {
      // around the towns: Basel, Bern, Genf, Zuerich
      result = R"( {
            "data": {
              "id": "Basel,Bern,Genf,Zuerich",
              "area": {
                    "type": "MultiPolygon",
                    "coordinates": [[
                    [ [47.54456,7.49078],
                      [47.46669,7.60892],
                      [47.47411,7.82322],
                      [47.57604,7.68859],
                      [47.60196,7.53474],
                      [47.54456,7.49078]
                    ],[
                      [47.60196,7.53474],
                      [47.57604,7.68859],
                      [47.47411,7.82322],
                      [47.46669,7.60892],
                      [47.54456,7.49078],
                      [47.04296,7.41119],
                      [46.96248,7.27382],
                      [47.60196,7.53474]
                    ],[
                      [46.13197,5.91837],
                      [46.22889,6.31125],
                      [46.41842,6.06948],
                      [46.54882,6.14916],
                      [46.62050,6.57501],
                      [46.49027,6.85799],
                      [46.34648,6.77282],
                      [46.13197,5.91837]
                    ],[
                      [47.53034,8.52952],
                      [47.51366,8.70810],
                      [47.36701,8.73832],
                      [47.31680,8.53501],
                      [47.38746,8.18060],
                      [47.48585,8.30972],
                      [47.53034,8.52952]
                    ]
                    ]]},
              "message": "This is a default message"
    }})";
    }
    else if(zone == 2) {
      // Wallis, Zürich und Luzern.
      result = R"( {
            "data": {
              "id": "west of Swiss",
              "area": {
                    "type": "MultiPolygon",
                    "coordinates": [[
                    [ [46.14554,6.92314],
                      [46.36946,7.31327],
                      [46.42624,7.71989],
                      [46.55851,8.42048],
                      [46.45461,8.44520],
                      [46.26710,8.07980],
                      [45.97604,7.72813],
                      [45.99321,7.01930],
                      [46.00939,7.81109],
                      [46.14554,6.92314]
                    ],[
                      [47.51569,8.36627],
                      [47.42478,8.14373],
                      [47.34674,7.85525],
                      [47.22572,7.83327],
                      [46.86290,7.84701],
                      [46.78591,8.49265],
                      [46.92292,8.92949],
                      [47.21454,9.04488],
                      [47.47648,8.88318],
                      [47.56730,8.57821],
                      [47.51569,8.36627]
                    ]
                    ]]},
              "message": "This is a default message"
    }})";
    }
    else if(zone == 3) {
      // Kantone Bern und Freiburg:
      result = R"( {
            "data": {
              "id": "east of Swiss",
              "area": {
                    "type": "MultiPolygon",
                    "coordinates": [[[
                      [46.65928,8.40749],
                      [46.83794,7.87998],
                      [47.22703,7.80031],
                      [47.07398,7.43216],
                      [46.37789,7.20412],
                      [46.54992,6.56672],
                      [46.50271,6.85520],
                      [46.79850,6.77827],
                      [46.98043,7.06400],
                      [47.22703,7.06518],
                      [46.65928,8.40749]
                      ]]]},
              "message": "This is a default message"
    }})";
    }
    return result;
  }
}

struct test_server::impl {
    impl(boost::asio::io_service& ios, const std::vector<std::string>& server_arguments)
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
      for(auto const& s : server_argv_) {
        //if(s == "medium") {
          //sleeping = 500;
        //}
        //else if(s == "high") {
          //sleeping = 1000;
        //}
        if(s == "1") {
          area = 1;
        }
        else if(s == "2") {
          area = 2;
        }
        else if(s == "3") {
          area = 3;
        }
        else if(s == "little") {
          on = true;
          fleet.resize(20);
          for(auto & i : fleet) {
            i.resize(24);
            for(int j = 0; j < i.size(); j++) {
              if(j < 2 || j > 4) {
                i.at(j) = 1;
              } else {
                i.at(j) = 0;
              }
            }
          }
        }
        else if(s == "normal") {
          on = true;
          fleet.resize(27);
          for(auto & i : fleet) {
            i.resize(24);
            for(int j = 0; j < i.size(); j++) {
              if(j < 2 || j > 4) {
                i.at(j) = 1;
              } else {
                i.at(j) = 0;
              }
            }
          }
        }
        else if(s == "big") {
          on = true;
          fleet.resize(35);
          for(auto & i : fleet) {
            i.resize(24);
            for(int j = 0; j < i.size(); j++) {
              if(j < 2 || j > 4) {
                i.at(j) = 1;
              } else {
                i.at(j) = 0;
              }
            }
          }
        }
      }
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


