#include "motis/intermodal/ondemand_availability.h"

//#include <rpc.h> // fuer die uuid
#include <ctime>

#include "boost/geometry.hpp"
#include "boost/geometry/geometries/point_xy.hpp"
#include "boost/geometry/geometries/polygon.hpp"
#include "boost/program_options.hpp"

#include "motis/core/common/logging.h"
#include "motis/core/common/constants.h"
#include "motis/core/common/timing.h"
#include "motis/module/context/motis_http_req.h"
#include "net/http/client/request.h"
#include "net/http/client/response.h"

#include "ctx/call.h"
#include "rapidjson/rapidjson.h"
#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "motis/json/json.h"
#include "date/date.h"

namespace opt = boost::program_options;

using namespace motis::module;
using namespace motis::json;
using namespace net::http::client;
using namespace ctx;
using namespace rapidjson;

namespace motis::intermodal {

availability_response read_result(response const& result, bool first, std::vector<geo::latlng> const& dots) {
    availability_response ares;
    if(result.status_code != 200) {
      ares.available = false;
      switch (result.status_code) {
        case 400: {
          LOG(logging::error) << "invalid inquiry "
                                 " http error code is: "
                              << result.status_code << "!"
                                 " Availability is set to false";
        }
        case 422: {
          LOG(logging::error) << " ride not available "
                                 " This went wrong: "
                              << result.body << "!"
                                 " Availability is set to false";
        }
        case 500: {
          LOG(logging::error) << " an unexpected http error occured "
                                 " http error code is: "
                              << result.status_code << "!"
                                 " Availability is set to false";
        }
        default:
          LOG(logging::error) << " something unexpected happened "
                                 " http error code is: "
                              << result.status_code << "!"
                                 " Availability is set to false";
     }
     return ares;
    }
    else {
    Document docu;
    if (docu.Parse(result.body.c_str()).HasParseError()) {
      docu.GetParseError();
      throw utl::fail("On-Demand Availability Check Response: Bad JSON: {} at offset {}",
                      GetParseError_En(docu.GetParseError()),
                      docu.GetErrorOffset());
    }
    auto const& data = get_obj(docu, "data");
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
    auto read_json_key_int = [&](char const* key, char const* name) -> int {
      auto const it = data.FindMember(key);
      if (it != data.MemberEnd() && it->value.IsInt()) {
        return it->value.GetInt();
      }
      else if(it != data.MemberEnd() && it->value.HasMember(name)) {
        auto const at = it->value.FindMember(name);
        if(at->value.IsInt()) {
          return at->value.GetInt();
        }
      }
      return -1;
    };
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
    auto read_json_key_array = [&](char const* key, char const* name) -> std::vector<std::vector<double>> {
      auto const it = data.FindMember(key);
      std::vector<std::vector<double>> vec;
      if(it != data.MemberEnd() && it->value.IsObject()) {
        auto const ar = it->value.FindMember(name);
        if(ar->value[0].Size() > 3) {
          vec.resize(ar->value[0].Size());
          for(SizeType a = 0; a < ar->value[0].Size(); a++) {
            const rapidjson::Value &data_vec = ar->value[0].GetArray()[a];
            for(SizeType b = 0; b < data_vec.Size(); b++) {
              vec[a].push_back(data_vec[b].GetDouble());
            }
          }
          return vec;
        } else {
          vec.resize(ar->value[0].GetArray()[0].Size());
          for(SizeType k = 0; k < ar->value[0].GetArray()[0].Size(); k++) {
            const rapidjson::Value &data_vec = ar->value[0].GetArray()[0].GetArray()[k];
            for(SizeType j = 0; j < data_vec.Size(); j++)
              vec[k].push_back(data_vec[j].GetDouble());
          }
          return vec;
        }
      }
      return vec;
    };

    if(first) {
      ares.codenumber_id = read_json_key_string("id", " ");
      std::vector<std::vector<double>> polypoints = read_json_key_array("area", "coordinates");
      std::vector<geo::latlng> polygon_area;
      polygon_area.resize(polypoints.size());
      int k = 0;
      for(auto const& vec : polypoints) {
        if(vec.size() == 2) {
          polygon_area[k].lat_ = vec.at(0);
          polygon_area[k].lng_ = vec.at(1);
        }
        else {
          LOG(logging::warn) << "invalid number of coordinates. "
                                "In http (get) result, with id: "
                             << ares.codenumber_id << "!"
                             << "availability is set to false";
          ares.available = false;
          return ares;
        }
        k++;
      }
      typedef boost::geometry::model::d2::point_xy<double> point_type;
      typedef boost::geometry::model::polygon<point_type> polygon_type;
      point_type point_one(dots.at(0).lat_, dots.at(0).lng_);
      point_type point_two(dots.at(1).lat_, dots.at(1).lng_);
      polygon_type poly;
      std::vector<geo::latlng>::iterator it;
      for(it = polygon_area.begin(); it != polygon_area.end(); it++) {
        geo::latlng dot = *it;
        boost::geometry::append(poly, boost::geometry::make<point_type>(dot.lat_, dot.lng_));
      }
      bool inside_start = boost::geometry::within(point_one, poly);
      bool inside_end = boost::geometry::within(point_two, poly);
      ares.available = inside_start && inside_end;
      return ares;
    }
    else {
      ares.codenumber_id = read_json_key_string("id", " ");
      ares.startpoint.lat_  = read_json_key_double("pickup", "lat");
      ares.startpoint.lng_ = read_json_key_double("pickup", "lng");
      ares.endpoint.lat_ = read_json_key_double("dropoff", "lat");
      ares.endpoint.lng_ = read_json_key_double("dropoff", "lng");
      ares.price = read_json_key_double("fare", "final_price");
      ares.walk_dur.emplace_back(read_json_key_int("pickup", "walking_duration"));
      ares.walk_dur.emplace_back(read_json_key_int("dropoff", "walking_duration"));
      std::string s_pickup_time = read_json_key_string("pickup", "negotiation_time");
      std::string s_dropoff_time = read_json_key_string("dropoff", "negotiation_time");
      // "2022-09-09T19:22:00Z" -> 1662751320
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
      ares.pickup_time = traveltime_to_unixtime(s_pickup_time).time_since_epoch().count();
      ares.dropoff_time = traveltime_to_unixtime(s_dropoff_time).time_since_epoch().count();
      return ares;
    }
  }
}

std::string create_json_body(availability_request const& areq) {
  // 1662751320 -> "2022-09-09T19:22:00Z"
  using time_point = std::chrono::system_clock::time_point;
  time_point time_convertion_departure{std::chrono::duration_cast<time_point::duration>(std::chrono::seconds(areq.departure_time))};
  time_point time_convertion_arrival{std::chrono::duration_cast<time_point::duration>(std::chrono::seconds(areq.arrival_time_onnext))};
  std::string dep_time = date::format("%FT%TZ", date::floor<std::chrono::seconds>(time_convertion_departure));
  std::string arr_time = date::format("%FT%TZ", date::floor<std::chrono::seconds>(time_convertion_arrival));

  // Creates a Ride Inquiry object with estimations and availability information - POST
  std::string json = R"( { "data": {
                      "product_id": ")" + areq.product_id + "\","
                + R"( "origin": {
                      "lat": )" + std::to_string(areq.startpoint.lat_) + ","
                + R"( "lng": )" + std::to_string(areq.startpoint.lng_) + ","
                + R"( "time": ")" + dep_time + "\""
                + R"(  }, "destination": {
                      "lat": )" + std::to_string(areq.endpoint.lat_) + ","
                + R"( "lng": )" + std::to_string(areq.endpoint.lng_) + ","
                + R"( "time": ")" + arr_time
                + "\"}}}";
  /*
   + "," + R"( "maxWalkDistance": )" + to_string(mars.maxWalkDist)
   + "}}";*/
  return json;
}

bool checking(availability_request const& areq, availability_response const& ares) {
  double delta = 0.00001;
  bool coord_start, coord_end, walklength, walktime, timewindow, waiting = true;
  bool result;
  if(areq.start) {
    //waiting = areq.departureTime + DELAY > ares.pickupTime[1] && // 1300 +15 = 1315 > 1310
    //          areq.departureTime - DELAY < ares.pickupTime[1]; // 1300 -15 = 1245 < 1310
    //printf("waiting: %lld + 15 > %lld \n -- %lld - 15 < %lld\n", areq.departureTime, ares.pickupTime[1], areq.departureTime, ares.pickupTime[1]);
    //printf("duration: %lld \n response duration complete: %lld \n", areq.duration,
     //(ares.dropoffTime[1] - ares.pickupTime[1]) + ares.walkDur.at(0) + ares.walkDur.at(1));
    if(ares.walk_dur.at(0) == 0 && ares.walk_dur.at(1) == 0) {
      coord_start = areq.startpoint.lat_ - ares.startpoint.lat_ < delta && areq.startpoint.lng_ - ares.startpoint.lng_ < delta;
      coord_end = areq.endpoint.lat_ - ares.endpoint.lat_ < delta && areq.endpoint.lng_ - ares.endpoint.lng_ < delta;
      timewindow = areq.duration >= ares.dropoff_time - ares.pickup_time;
      //printf("checking start: waiting: %d; coord_start: %d; coord_end: %d; timewindow: %d\n", waiting, coord_start, coord_end, timewindow);
      result = coord_start && coord_end && waiting && timewindow;
    }
    else if(ares.walk_dur.at(0) != 0 && ares.walk_dur.at(1) == 0) {
      walktime = ares.walk_dur.at(0) < MAX_WALK_TIME;
      walklength = areq.max_walk_dist >= ares.walk_dur.at(0) * WALK_SPEED;
      timewindow = areq.duration >= (ares.dropoff_time - ares.pickup_time) + ares.walk_dur.at(0);
      //printf("checking: waiting: %d; walktime: %d; walklength: %d; timewindow: %d\n", waiting, walktime, walklength, timewindow);
      result = walklength && walktime && waiting && timewindow;
    }
    else if(ares.walk_dur.at(1) != 0 && ares.walk_dur.at(0) == 0) {
      walktime = ares.walk_dur.at(1) < MAX_WALK_TIME &&
                 ares.dropoff_time + ares.walk_dur.at(1) < areq.arrival_time_onnext; // 1350 +5 = 1355 < 1400
      walklength = areq.max_walk_dist >= ares.walk_dur.at(1) * WALK_SPEED;
      timewindow = areq.duration >= (ares.dropoff_time - ares.pickup_time) + ares.walk_dur.at(1);
      //printf("checking: waiting: %d; walktime: %d; walklength: %d; timewindow: %d\n", waiting, walktime, walklength, timewindow);
      result = walklength && walktime && waiting && timewindow;
    }
    else {
      walktime = ares.walk_dur.at(0) < MAX_WALK_TIME && ares.walk_dur.at(1) < MAX_WALK_TIME
                 && ares.dropoff_time + ares.walk_dur.at(1) < areq.arrival_time_onnext;
      walklength = areq.max_walk_dist >= ares.walk_dur.at(1) * WALK_SPEED && areq.max_walk_dist >= ares.walk_dur.at(0) * WALK_SPEED;
      timewindow = areq.duration >= (ares.dropoff_time - ares.pickup_time) + ares.walk_dur.at(0) + ares.walk_dur.at(1);
      //printf("checking: waiting: %d; walktime: %d; walklength: %d; timewindow: %d\n", waiting, walktime, walklength, timewindow);
      result = walklength && walktime && waiting && timewindow;
    }
  } else {
    //waiting = areq.departureTime + DELAY > ares.pickupTime[1] // 1300 +15 = 1315 > 1310
    //          && areq.arrivalTime < ares.pickupTime[1];       // 1258 < 1310
    //printf("waiting: %lld + 15 > %lld \n -- %lld < %lld\n", areq.departureTime, ares.pickupTime[1], areq.arrivalTime, ares.pickupTime[1]);
    if(ares.walk_dur.at(0) == 0 && ares.walk_dur.at(1) == 0) {
      coord_start = areq.startpoint.lat_ - ares.startpoint.lat_ < delta && areq.startpoint.lng_ - ares.startpoint.lng_ < delta;
      coord_end = areq.endpoint.lat_ - ares.endpoint.lat_ < delta && areq.endpoint.lng_ - ares.endpoint.lng_ < delta;
      timewindow = areq.duration >= ares.dropoff_time - ares.pickup_time;
      //printf("checking end: waiting: %d; coord_start: %d; coord_end: %d; timewindow: %d\n", waiting, coord_start, coord_end, timewindow);
      result = coord_start && coord_end && waiting && timewindow;
    }
    else if(ares.walk_dur.at(0) != 0 && ares.walk_dur.at(1) == 0) {
      walktime = areq.departure_time + ares.walk_dur.at(0) < ares.pickup_time && ares.walk_dur.at(0) < MAX_WALK_TIME; // 1300 +5 = 1305 < 1310
      walklength = areq.max_walk_dist >= ares.walk_dur.at(0) * WALK_SPEED;
      timewindow = areq.duration >= (ares.dropoff_time - ares.pickup_time) + ares.walk_dur.at(0);
      //printf("walk: %lld + %d < %lld && %d < %d \n", areq.departureTime, ares.walkDur.at(0), ares.pickupTime[1], ares.walkDur.at(0), MAX_WALK_TIME);
      //printf("checking: waiting: %d; walktime: %d; walklength: %d; timewindow: %d\n", waiting, walktime, walklength, timewindow);
      result = walklength && walktime && waiting && timewindow;
    }
    else if(ares.walk_dur.at(1) != 0 && ares.walk_dur.at(0) == 0) {
      walktime = ares.walk_dur.at(1) < MAX_WALK_TIME;
      walklength = areq.max_walk_dist >= ares.walk_dur.at(1) * WALK_SPEED;
      timewindow = areq.duration >= (ares.dropoff_time - ares.pickup_time) + ares.walk_dur.at(1); // evtl egal -> dann länge der kante anpassen ?
      //printf("checking: waiting: %d; walktime: %d; walklength: %d; timewindow: %d\n", waiting, walktime, walklength, timewindow);
      result = walklength && walktime && waiting && timewindow;
    }
    else {
      walktime = ares.walk_dur.at(0) < MAX_WALK_TIME && ares.walk_dur.at(1) < MAX_WALK_TIME
                 && areq.departure_time + ares.walk_dur.at(0) < ares.pickup_time;
      walklength = areq.max_walk_dist >= ares.walk_dur.at(1) * WALK_SPEED && areq.max_walk_dist >= ares.walk_dur.at(0) * WALK_SPEED;
      timewindow = areq.duration >= (ares.dropoff_time - ares.pickup_time) + ares.walk_dur.at(0) + ares.walk_dur.at(1);
      //printf("checking: waiting: %d; walktime: %d; walklength: %d; timewindow: %d\n", waiting, walktime, walklength, timewindow);
      result = walklength && walktime && waiting && timewindow;
    }
  }
  return result;
}

availability_response check_od_availability(availability_request areq,
                                            std::vector<std::string> const& server_infos,
                                            statistics& stats) {
  std::string addr;
  std::string second_addr;
  std::map<std::string, std::string> hdrs;
  for(auto const& info : server_infos) {
    size_t index = info.find(':');
    if(index == -1) {
      size_t idx = info.find(',');
      hdrs.insert(std::pair<std::string, std::string>(info.substr(0, idx), info.substr(idx+1)));
    }
    std::string name = info.substr(0, index);
    if(name == "address") {
      addr = info.substr(index+1);
    }
    else if(name == "address2") {
      second_addr = info.substr(index+1);
    }
    else if(name == "productid") {
      areq.product_id = info.substr(index+1);
    }
  }

  request::method m_get = request::GET;
  request req(addr, m_get, hdrs, "");

  geo::latlng req_dot_start;
  geo::latlng req_dot_end;
  req_dot_end.lat_ = areq.endpoint.lat_;
  req_dot_end.lng_ = areq.endpoint.lng_;
  req_dot_start.lat_ = areq.startpoint.lat_;
  req_dot_start.lng_ = areq.startpoint.lng_;
  std::vector<geo::latlng> req_dots;
  req_dots.emplace_back(req_dot_start);
  req_dots.emplace_back(req_dot_end);

  MOTIS_START_TIMING(ondemand_server_first);
  response firstresult = motis_http(req)->val();
  MOTIS_STOP_TIMING(ondemand_server_first);
  stats.ondemand_server_first_inquery_ +=
      static_cast<uint64_t>(MOTIS_TIMING_MS(ondemand_server_first));

  availability_response response_first = read_result(firstresult, true, req_dots);
  if(!response_first.available) {
    return response_first;
  }
  else {
    request::method m_post = request::POST;
    //UUID uuid;
    //UuidCreate(&uuid);
    //char* random_uuid_str;
    //UuidToStringA(&uuid, (RPC_CSTR*)&random_uuid_str);
    //hdrs.insert(pair<string, string>("Idempotency-Key", random_uuid_str));
    std::string body = create_json_body(areq);
    request req2(second_addr, m_post, hdrs, body);

    MOTIS_START_TIMING(ondemand_server_second);
    response secondresult = motis_http(req2)->val();
    MOTIS_STOP_TIMING(ondemand_server_second);
    stats.ondemand_server_second_inquery_ +=
        static_cast<uint64_t>(MOTIS_TIMING_MS(ondemand_server_second));

    availability_response response_second = read_result(secondresult, false, req_dots);
    response_second.available = checking(areq, response_second);
    return response_second;
  }
}

} // namespace intermodal