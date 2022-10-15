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
      ares.available_ = false;
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
      if(docu.GetParseError() == rapidjson::kParseErrorDocumentEmpty) {
        ares.available_ = false;
        return ares;
      }
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
      ares.codenumber_id_ = read_json_key_string("id", " ");
      std::vector<std::vector<double>> polypoints = read_json_key_array("area", "coordinates");
      if(polypoints.empty()) {
        ares.available_ = false;
        return ares;
      }
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
                             << ares.codenumber_id_ << "!"
                             << "availability is set to false";
          ares.available_ = false;
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
      ares.available_ = inside_start && inside_end;
      return ares;
    }
    else {
      ares.codenumber_id_ = read_json_key_string("id", " ");
      ares.startpoint_.lat_  = read_json_key_double("pickup", "lat");
      ares.startpoint_.lng_ = read_json_key_double("pickup", "lng");
      ares.endpoint_.lat_ = read_json_key_double("dropoff", "lat");
      ares.endpoint_.lng_ = read_json_key_double("dropoff", "lng");
      ares.price_ = read_json_key_double("fare", "final_price");
      ares.walk_dur_.at(0) = read_json_key_int("pickup", "walking_duration");
      ares.walk_dur_.at(1) = read_json_key_int("dropoff", "walking_duration");
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
      ares.pickup_time_ = traveltime_to_unixtime(s_pickup_time).time_since_epoch().count();
      ares.dropoff_time_ = traveltime_to_unixtime(s_dropoff_time).time_since_epoch().count();
      ares.complete_duration_ = (ares.dropoff_time_ - ares.pickup_time_) + ares.walk_dur_.at(0) + ares.walk_dur_.at(1);
      return ares;
    }
  }
}

std::string create_json_body(availability_request const& areq) {
  // 1662751320 -> "2022-09-09T19:22:00Z"
  using time_point = std::chrono::system_clock::time_point;
  time_point time_convertion_departure{std::chrono::duration_cast<time_point::duration>(std::chrono::seconds(areq.departure_time_))};
  time_point time_convertion_arrival{std::chrono::duration_cast<time_point::duration>(std::chrono::seconds(areq.arrival_time_onnext_))};
  std::string dep_time = date::format("%FT%TZ", date::floor<std::chrono::seconds>(time_convertion_departure));
  std::string arr_time = date::format("%FT%TZ", date::floor<std::chrono::seconds>(time_convertion_arrival));
  // Creates a Ride Inquiry object with estimations and availability information - POST
  std::string json = R"( { "data": {
                      "product_id": ")" + areq.product_id_ + "\","
                + R"( "origin": {
                      "lat": )" + std::to_string(areq.startpoint_.lat_) + ","
                + R"( "lng": )" + std::to_string(areq.startpoint_.lng_) + ","
                + R"( "time": ")" + dep_time + "\""
                + R"(  }, "destination": {
                      "lat": )" + std::to_string(areq.endpoint_.lat_) + ","
                + R"( "lng": )" + std::to_string(areq.endpoint_.lng_) + ","
                + R"( "time": ")" + arr_time
                + "\"}}}";
  /*
   + "," + R"( "maxWalkDistance": )" + to_string(mars.maxWalkDist)
   + "}}";*/
  return json;
}

bool checking(availability_request const& areq, availability_response const& ares) {
  double epsilon = 0.00001;
  unixtime delta = 900;
  bool coord_start = false, coord_end = false, walklength = false, walktime = false, timewindow;
  bool result = false;
  unixtime timediff_dep = abs(areq.departure_time_ - ares.pickup_time_);
  unixtime timediff_arr = abs(areq.arrival_time_onnext_ - ares.dropoff_time_);
  if(timediff_dep > delta || timediff_arr > delta) {
    return false;
  }
  timewindow = areq.duration_ >= ares.complete_duration_;
  if(areq.start_) {
    if(ares.walk_dur_.at(0) == 0 && ares.walk_dur_.at(1) == 0) {
      coord_start = areq.startpoint_.lat_ - ares.startpoint_.lat_ < epsilon && areq.startpoint_.lng_ - ares.startpoint_.lng_ < epsilon;
      coord_end = areq.endpoint_.lat_ - ares.endpoint_.lat_ < epsilon && areq.endpoint_.lng_ - ares.endpoint_.lng_ < epsilon;
      result = coord_start & coord_end & timewindow;
    }
    else if(ares.walk_dur_.at(0) != 0 && ares.walk_dur_.at(1) == 0) {
      walktime = ares.walk_dur_.at(0) < MAX_WALK_TIME;
      walklength = areq.max_walk_dist_ >= ares.walk_dur_.at(0) * WALK_SPEED;
      result = walklength & walktime & timewindow;
    }
    else if(ares.walk_dur_.at(1) != 0 && ares.walk_dur_.at(0) == 0) {
      walktime = ares.walk_dur_.at(1) < MAX_WALK_TIME &&
                 ares.dropoff_time_ + ares.walk_dur_.at(1) < areq.arrival_time_onnext_; // 1350 +5 = 1355 < 1400
      walklength = areq.max_walk_dist_ >= ares.walk_dur_.at(1) * WALK_SPEED;
      result = walklength & walktime & timewindow;
    }
    else {
      walktime = ares.walk_dur_.at(0) < MAX_WALK_TIME && ares.walk_dur_.at(1) < MAX_WALK_TIME
                 && ares.dropoff_time_ + ares.walk_dur_.at(1) < areq.arrival_time_onnext_;
      walklength = areq.max_walk_dist_ >= ares.walk_dur_.at(1) * WALK_SPEED && areq.max_walk_dist_ >= ares.walk_dur_.at(0) * WALK_SPEED;
      result = walklength & walktime & timewindow;
    }
  } else {
    if(ares.walk_dur_.at(0) == 0 && ares.walk_dur_.at(1) == 0) {
      coord_start = areq.startpoint_.lat_ - ares.startpoint_.lat_ < epsilon && areq.startpoint_.lng_ - ares.startpoint_.lng_ < epsilon;
      coord_end = areq.endpoint_.lat_ - ares.endpoint_.lat_ < epsilon && areq.endpoint_.lng_ - ares.endpoint_.lng_ < epsilon;
      result = coord_start & coord_end & timewindow;
    }
    else if(ares.walk_dur_.at(0) != 0 && ares.walk_dur_.at(1) == 0) {
      walktime = areq.departure_time_ + ares.walk_dur_.at(0) < ares.pickup_time_ && ares.walk_dur_.at(0) < MAX_WALK_TIME; // 1300 +5 = 1305 < 1310
      walklength = areq.max_walk_dist_ >= ares.walk_dur_.at(0) * WALK_SPEED;
      result = walklength & walktime & timewindow;
    }
    else if(ares.walk_dur_.at(1) != 0 && ares.walk_dur_.at(0) == 0) {
      walktime = ares.walk_dur_.at(1) < MAX_WALK_TIME;
      walklength = areq.max_walk_dist_ >= ares.walk_dur_.at(1) * WALK_SPEED;
      result = walklength & walktime & timewindow;
    }
    else {
      walktime = ares.walk_dur_.at(0) < MAX_WALK_TIME && ares.walk_dur_.at(1) < MAX_WALK_TIME
                 && areq.departure_time_ + ares.walk_dur_.at(0) < ares.pickup_time_;
      walklength = areq.max_walk_dist_ >= ares.walk_dur_.at(1) * WALK_SPEED && areq.max_walk_dist_ >= ares.walk_dur_.at(0) * WALK_SPEED;
      result = walklength & walktime & timewindow;
    }
  }
  return result;
}

bool check_od_area(geo::latlng from, geo::latlng to,
                   std::vector<std::string> const& server_infos,
                   statistics& stats) {
  std::string area_check_addr;
  std::map<std::string, std::string> hdrs;
  hdrs.insert(std::pair<std::string, std::string>("Accept","application/json"));
  hdrs.insert(std::pair<std::string, std::string>("Accept-Language","de"));
  for(auto const& info : server_infos) {
    size_t index = info.find(':');
    if(index == -1) {
      size_t idx = info.find(',');
      hdrs.insert(std::pair<std::string, std::string>(info.substr(0, idx), info.substr(idx+1)));
    }
    std::string name = info.substr(0, index);
    if(name == "address") {
      area_check_addr = info.substr(index+1);
    }
  }

  if(area_check_addr != "http://127.0.0.1:9000/") {
    hdrs.insert(std::pair<std::string, std::string>("Content-Type","application/json"));
    hdrs.insert(std::pair<std::string, std::string>("X-Client-Version","0.0.1"));
    hdrs.insert(std::pair<std::string, std::string>("X-Api-Version","20210101"));
    hdrs.insert(std::pair<std::string, std::string>("X-Client-Identifier","demo.motis.rmv.platform"));
  }
  request::method m_get = request::GET;
  request area_check_req(area_check_addr, m_get, hdrs, "");
  std::vector<geo::latlng> req_dots;
  req_dots.emplace_back(from);
  req_dots.emplace_back(to);

  MOTIS_START_TIMING(ondemand_server_area);
  response area_check_result = motis_http(area_check_req)->val();
  MOTIS_STOP_TIMING(ondemand_server_area);
  stats.ondemand_server_area_inquery_ +=
      static_cast<uint64_t>(MOTIS_TIMING_MS(ondemand_server_area));
  availability_response area_check_response = read_result(area_check_result, true, req_dots);
  return area_check_response.available_;
}

void check_od_availability(const availability_request& areq,
                                            statistics& stats,
                                            std::vector<availability_response>& vares) {
  request::method m_post = request::POST;
  std::vector<geo::latlng> req_dots{};
  //UUID uuid;
  //UuidCreate(&uuid);
  //char* random_uuid_str;
  //UuidToStringA(&uuid, (RPC_CSTR*)&random_uuid_str);
  //hdrs.insert(pair<string, string>("Idempotency-Key", random_uuid_str));
  std::string body = create_json_body(areq);
  request product_check_req(areq.product_check_addr_, m_post, areq.hdrs_, body);
  auto f_product_check = http_future_t{};

  MOTIS_START_TIMING(ondemand_server_product);
  f_product_check = motis_http(product_check_req);
  MOTIS_STOP_TIMING(ondemand_server_product);
  stats.ondemand_server_product_inquery_ +=
      static_cast<uint64_t>(MOTIS_TIMING_MS(ondemand_server_product));

  availability_response product_check_response = read_result(f_product_check->val(), false, req_dots);
  product_check_response.available_ = checking(areq, product_check_response);
  product_check_response.journey_id_ = areq.journey_id_;
  auto const lock = std::scoped_lock{lock_vares_};
  vares.emplace_back(product_check_response);
}

} // namespace intermodal