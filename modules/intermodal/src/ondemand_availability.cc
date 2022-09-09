#include "motis/intermodal/ondemand_availability.h"

//#include <rpc.h> // fuer die uuid
#include <ctime>

#include "boost/geometry.hpp"
#include "boost/geometry/geometries/point_xy.hpp"
#include "boost/geometry/geometries/polygon.hpp"
#include "boost/program_options.hpp"

#include "motis/core/common/unixtime.h"
#include "motis/core/common/logging.h"
#include "motis/core/common/constants.h"
#include "motis/module/context/motis_http_req.h"
#include "net/http/client/request.h"
#include "net/http/client/response.h"

#include "ctx/call.h"
#include "rapidjson/rapidjson.h"
#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "motis/json/json.h"

namespace opt = boost::program_options;

using namespace motis::module;
using namespace motis::json;
using namespace net::http::client;
using namespace ctx;
using namespace rapidjson;

namespace motis::intermodal {
#define DELAY 900  // 15min

availability_response read_result(const response& result, bool first, std::vector<geo::latlng> dots)
{
  //printf("read_result: \n");
  availability_response ares;
  if(result.status_code != 200)
  {
    ares.available = false;
    if(result.status_code == 400)
    {
      LOG(logging::error) << "invalid inquiry "
                            " http error code is: "
                         << result.status_code << "!"
                            "availability is set to false";
    }
    else if(result.status_code == 422)
    {
      LOG(logging::error) << " ride not available "
                             " This went wrong: "
                          << result.body << "!"
                             "availability is set to false";
    }
    else if(result.status_code == 500)
    {
      LOG(logging::error) << " an unexpected http error occured "
                             " http error code is: "
                          << result.status_code << "!"
                             "availability is set to false";
    }
    else
    {
      LOG(logging::error) << " something unexpected happened "
                             " http error code is: "
                          << result.status_code << "!"
                             "availability is set to false";
    }
    return ares;
  }
  else
  {
    Document docu;
    if (docu.Parse(result.body.c_str()).HasParseError())
    {
      docu.GetParseError();
      throw utl::fail("On-Demand Availability Check Response: Bad JSON: {} at offset {}",
                      GetParseError_En(docu.GetParseError()),
                      docu.GetErrorOffset());
    }
    auto const& data = get_obj(docu, "data");
    auto read_json_key_string = [&](char const* key, char const* name) -> std::string
    {
      auto const it = data.FindMember(key);
      if (it != data.MemberEnd() && it->value.IsString())
      {
        return it->value.GetString();
      }
      else if(it != data.MemberEnd() && it->value.HasMember(name))
      {
        auto const at = it->value.FindMember(name);
        if(at->value.IsString())
        {
          return at->value.GetString();
        }
      }
      return "";
    };
    auto read_json_key_int = [&](char const* key, char const* name) -> int
    {
      auto const it = data.FindMember(key);
      if (it != data.MemberEnd() && it->value.IsInt())
      {
        return it->value.GetInt();
      }
      else if(it != data.MemberEnd() && it->value.HasMember(name))
      {
        auto const at = it->value.FindMember(name);
        if(at->value.IsInt())
        {
          return at->value.GetInt();
        }
      }
      return -1;
    };
    auto read_json_key_double = [&](char const* key, char const* name) -> double
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
    auto read_json_key_array = [&](char const* key, char const* name) -> std::vector<std::vector<double>>
    {
      auto const it = data.FindMember(key);
      std::vector<std::vector<double>> vec;
      if (it != data.MemberEnd() && it->value.IsObject())
      {
        auto const ar = it->value.FindMember(name);
        //Gesamtarray: wenn so aufgebaut wie angegeben
        //ar->value[0].GetArray()[0]
        //ar->value[0].GetArray()[0].Size()
        //erstes Koordinatenarray
        //ar->value[0].GetArray()[0].GetArray()[0]
        //Inhalt des ersten Koordinatenarrays (Koordinate 0)
        //ar->value[0].GetArray()[0].GetArray()[0].GetArray()[0]
        // ansonsten -> ohne zusaetzliche Klammern
        if(ar->value[0].Size() > 3)
        {
          vec.resize(ar->value[0].Size());
          for(SizeType a = 0; a < ar->value[0].Size(); a++)
          {
            const rapidjson::Value &data_vec = ar->value[0].GetArray()[a];
            for(SizeType b = 0; b < data_vec.Size(); b++)
            {
              vec[a].push_back(data_vec[b].GetDouble());
            }
          }
          return vec;
        }
        else
        {
          vec.resize(ar->value[0].GetArray()[0].Size());
          for(SizeType k = 0; k < ar->value[0].GetArray()[0].Size(); k++)
          {
            const rapidjson::Value &data_vec = ar->value[0].GetArray()[0].GetArray()[k];
            for(SizeType j = 0; j < data_vec.Size(); j++)
              vec[k].push_back(data_vec[j].GetDouble());
          }
          return vec;
        }
      }
      return vec;
    };

    if(first)
    {
      ares.codenumber_id = read_json_key_string("id", " ");
      std::vector<std::vector<double>> polypoints = read_json_key_array("area", "coordinates");
      std::vector<geo::latlng> polygon_area;
      polygon_area.resize(polypoints.size());
      int k = 0;
      for(auto const& vec : polypoints)
      {
        if(vec.size() == 2)
        {
          polygon_area[k].lat_ = vec.at(0);
          polygon_area[k].lng_ = vec.at(1);
        }
        else
        {
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
      for(it = polygon_area.begin(); it != polygon_area.end(); it++)
      {
        geo::latlng dot = *it;
        boost::geometry::append(poly, boost::geometry::make<point_type>(dot.lat_, dot.lng_));
      }
      bool inside_start = boost::geometry::within(point_one, poly);
      bool inside_end = boost::geometry::within(point_two, poly);
      ares.available = inside_start && inside_end;
      return ares;
    }
    else
    {
      ares.codenumber_id = read_json_key_string("id", " ");
      ares.startpoint.lat_  = read_json_key_double("pickup", "lat");
      ares.startpoint.lng_ = read_json_key_double("pickup", "lng");
      ares.endpoint.lat_ = read_json_key_double("dropoff", "lat");
      ares.endpoint.lng_ = read_json_key_double("dropoff", "lng");
      ares.price = read_json_key_double("fare", "final_price");
      ares.walkDur.emplace_back(read_json_key_int("pickup", "walking_duration"));
      ares.walkDur.emplace_back(read_json_key_int("dropoff", "walking_duration"));
      std::string pu_time2 = read_json_key_string("pickup", "negotiation_time");
      std::string do_time2 = read_json_key_string("dropoff", "negotiation_time");
      //"2017-09-06T15:13:43Z" -> 1504703623
      auto traveltime_to_unixtime = [&](const std::string& timestring) -> int64_t
      {
        int year = stoi(timestring.substr(0, 4));
        int month = stoi(timestring.substr(5, 2));
        int day = stoi(timestring.substr(8, 2));
        size_t pos_T = timestring.find('T');
        std::string str = timestring.substr(pos_T + 1, 8);
        str.erase(2,1);
        str.erase(4,1);
        int hour = stoi(str.substr(0, 2));
        int min = stoi(str.substr(2, 2));
        int sec = stoi(str.substr(4, 2));
        boost::gregorian::date firstswitchday(year, month, day);
        boost::gregorian::date secondswitchday(year, month, day);
        for(int i = 25; i < 32; i++)
        {
          boost::gregorian::date::year_type y = year;
          boost::gregorian::date::day_type d = i;
          boost::gregorian::date date(y, 3, d);
          if(0 == date.day_of_week()) // 0 Sonntag ?
          {
            firstswitchday = {y, 3, d};
          }
          boost::gregorian::date date2(y, 10, d);
          if(0 == date2.day_of_week())
          {
            secondswitchday = {y, 10, d};
          }
        }
        boost::gregorian::date currentday(year, month, day);
        struct tm name = {0};
        name.tm_year = year - 1900;
        name.tm_mon = month - 1;
        name.tm_mday = day;
        name.tm_hour = hour;
        name.tm_min = min;
        name.tm_sec = sec;
        if(currentday >= firstswitchday && currentday <= secondswitchday && hour > 2)
        {
          // march until october
          name.tm_isdst = 1;
        }
        else if(currentday >= secondswitchday && currentday <= firstswitchday && hour > 2)
        {
          // october until march
          name.tm_isdst = 0;
        }
        time_t result = mktime(&name);
        return result;
      };
      ares.pickupTime = traveltime_to_unixtime(pu_time2);
      ares.dropoffTime = traveltime_to_unixtime(do_time2);
      return ares;
    }
  }
}

std::string create_json_body(const availability_request& areq)
{
  // 1504703623 -> "2017-09-06T15:13:43Z"
  auto unixtime_to_traveltime = [&](const unixtime& timeunix) -> std::string
  {
    std::string result;
    time_t thistime = static_cast<time_t>(timeunix);
    struct tm tm_info = {0};
    localtime_s(&tm_info, &thistime);
    std::string month = std::to_string(tm_info.tm_mon + 1);
    std::string day = std::to_string(tm_info.tm_mday);
    std::string hour = std::to_string(tm_info.tm_hour);
    std::string minutes = std::to_string(tm_info.tm_min);
    std::string seconds = std::to_string(tm_info.tm_sec);
    std::string year = std::to_string(tm_info.tm_year + 1900);
    if(tm_info.tm_mon < 10) month = "0" + std::to_string(tm_info.tm_mon);
    if(tm_info.tm_mday < 10) day = "0" + std::to_string(tm_info.tm_mday);
    if(tm_info.tm_hour < 10) hour = "0" + std::to_string(tm_info.tm_hour);
    if(tm_info.tm_min < 10) minutes = "0" + std::to_string(tm_info.tm_min);
    if(tm_info.tm_sec < 10) seconds = "0" + std::to_string(tm_info.tm_sec);
    result = year + "-" + month + "-" + day + "T" + hour + ":" + minutes + ":"
               + seconds + "Z";
    return result;
  };

  std::string dep_time = unixtime_to_traveltime(areq.departureTime);
  std::string arr_time = unixtime_to_traveltime(areq.arrivalTime_onnext);
  // Creates a Ride Inquiry object with estimations and availability information - POST
  std::string json = R"( { "data": {
                      "product_id": ")" + areq.productID + "\","
                + R"( "origin": {
                      "lat": )" + std::to_string(areq.startpoint.lat_) + ","
                + R"( "lng": )" + std::to_string(areq.startpoint.lng_) + ","
                + R"( "time": ")" + dep_time + "\","
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

bool checking(const availability_request& areq, const availability_response& ares)
{
  auto coord_equality = [&](const std::string& eins, const std::string& zwei) -> bool
  {
    size_t lastidx1 = eins.length() - 1;
    size_t lastidx2 = zwei.length() - 1;
    size_t pidx1 = eins.find('.');
    size_t pidx2 = zwei.find('.');
    std::string one = eins.substr(0, lastidx1);
    std::string oneverify = one.substr(pidx1 + 1, one.length() - 1 - pidx1);
    std::string two = zwei.substr(0, lastidx2);
    std::string twoverify = two.substr(pidx2 + 1, two.length() - 1 - pidx2);
    if(oneverify.length() != twoverify.length())
    {
      if(oneverify.length() > twoverify.length())
      {
        if(twoverify.length() < 4) return false;
        size_t diff = oneverify.length() - twoverify.length();
        std::string newone = one.substr(one.length() - 1 - diff, diff);
        return newone == two;
      }
      else
      {
        if(oneverify.length() < 4) return false;
        size_t diff = twoverify.length() - oneverify.length();
        std::string newtwo = two.substr(two.length() - 1 - diff, diff);
        return newtwo == one;
      }
    }
    else
    {
      return one == two;
    }
  };
  bool coord_start, coord_end, walklength, walktime, timewindow, waiting = true;
  bool result;
  if(areq.start)
  {
    //waiting = areq.departureTime + DELAY > ares.pickupTime[1] && // 1300 +15 = 1315 > 1310
    //          areq.departureTime - DELAY < ares.pickupTime[1]; // 1300 -15 = 1245 < 1310
    //printf("waiting: %lld + 15 > %lld \n -- %lld - 15 < %lld\n", areq.departureTime, ares.pickupTime[1], areq.departureTime, ares.pickupTime[1]);
    //printf("duration: %lld \n response duration complete: %lld \n", areq.duration,
     //(ares.dropoffTime[1] - ares.pickupTime[1]) + ares.walkDur.at(0) + ares.walkDur.at(1));
    if(ares.walkDur.at(0) == 0 && ares.walkDur.at(1) == 0)
    {
      coord_start = coord_equality(std::to_string(areq.startpoint.lat_), std::to_string(ares.startpoint.lat_))
                    && coord_equality(std::to_string(areq.startpoint.lng_), std::to_string(ares.startpoint.lng_));
      coord_end = coord_equality(std::to_string(areq.endpoint.lat_), std::to_string(ares.endpoint.lat_)) &&
                  coord_equality(std::to_string(areq.endpoint.lng_), std::to_string(ares.endpoint.lng_));
      timewindow = areq.duration >= ares.dropoffTime - ares.pickupTime;
      //printf("checking start: waiting: %d; coord_start: %d; coord_end: %d; timewindow: %d\n", waiting, coord_start, coord_end, timewindow);
      result = coord_start && coord_end && waiting && timewindow;
    }
    else if(ares.walkDur.at(0) != 0 && ares.walkDur.at(1) == 0)
    {
      walktime = ares.walkDur.at(0) < MAX_WALK_TIME;
      walklength = areq.maxWalkDist >= ares.walkDur.at(0) * WALK_SPEED;
      timewindow = areq.duration >= (ares.dropoffTime - ares.pickupTime) + ares.walkDur.at(0);
      //printf("checking: waiting: %d; walktime: %d; walklength: %d; timewindow: %d\n", waiting, walktime, walklength, timewindow);
      result = walklength && walktime && waiting && timewindow;
    }
    else if(ares.walkDur.at(1) != 0 && ares.walkDur.at(0) == 0)
    {
      walktime = ares.walkDur.at(1) < MAX_WALK_TIME &&
                 ares.dropoffTime + ares.walkDur.at(1) < areq.arrivalTime_onnext; // 1350 +5 = 1355 < 1400
      walklength = areq.maxWalkDist >= ares.walkDur.at(1) * WALK_SPEED;
      timewindow = areq.duration >= (ares.dropoffTime - ares.pickupTime) + ares.walkDur.at(1);
      //printf("checking: waiting: %d; walktime: %d; walklength: %d; timewindow: %d\n", waiting, walktime, walklength, timewindow);
      result = walklength && walktime && waiting && timewindow;
    }
    else
    {
      walktime = ares.walkDur.at(0) < MAX_WALK_TIME && ares.walkDur.at(1) < MAX_WALK_TIME
                 && ares.dropoffTime + ares.walkDur.at(1) < areq.arrivalTime_onnext;
      walklength = areq.maxWalkDist >= ares.walkDur.at(1) * WALK_SPEED && areq.maxWalkDist >= ares.walkDur.at(0) * WALK_SPEED;
      timewindow = areq.duration >= (ares.dropoffTime - ares.pickupTime) + ares.walkDur.at(0) + ares.walkDur.at(1);
      //printf("checking: waiting: %d; walktime: %d; walklength: %d; timewindow: %d\n", waiting, walktime, walklength, timewindow);
      result = walklength && walktime && waiting && timewindow;
    }
  }
  else
  {
    //waiting = areq.departureTime + DELAY > ares.pickupTime[1] // 1300 +15 = 1315 > 1310
    //          && areq.arrivalTime < ares.pickupTime[1];       // 1258 < 1310
    //printf("waiting: %lld + 15 > %lld \n -- %lld < %lld\n", areq.departureTime, ares.pickupTime[1], areq.arrivalTime, ares.pickupTime[1]);
    if(ares.walkDur.at(0) == 0 && ares.walkDur.at(1) == 0)
    {
      coord_start = coord_equality(std::to_string(areq.startpoint.lat_), std::to_string(ares.startpoint.lat_)) &&
                    coord_equality(std::to_string(areq.startpoint.lng_), std::to_string(ares.startpoint.lng_));
      coord_end = coord_equality(std::to_string(areq.endpoint.lat_), std::to_string(ares.endpoint.lat_)) &&
                  coord_equality(std::to_string(areq.endpoint.lng_), std::to_string(ares.endpoint.lng_));
      timewindow = areq.duration >= ares.dropoffTime - ares.pickupTime;
      //printf("checking end: waiting: %d; coord_start: %d; coord_end: %d; timewindow: %d\n", waiting, coord_start, coord_end, timewindow);
      result = coord_start && coord_end && waiting && timewindow;
    }
    else if(ares.walkDur.at(0) != 0 && ares.walkDur.at(1) == 0)
    {
      walktime = areq.departureTime + ares.walkDur.at(0) < ares.pickupTime && ares.walkDur.at(0) < MAX_WALK_TIME; // 1300 +5 = 1305 < 1310
      walklength = areq.maxWalkDist >= ares.walkDur.at(0) * WALK_SPEED;
      timewindow = areq.duration >= (ares.dropoffTime - ares.pickupTime) + ares.walkDur.at(0);
      //printf("walk: %lld + %d < %lld && %d < %d \n", areq.departureTime, ares.walkDur.at(0), ares.pickupTime[1], ares.walkDur.at(0), MAX_WALK_TIME);
      //printf("checking: waiting: %d; walktime: %d; walklength: %d; timewindow: %d\n", waiting, walktime, walklength, timewindow);
      result = walklength && walktime && waiting && timewindow;
    }
    else if(ares.walkDur.at(1) != 0 && ares.walkDur.at(0) == 0)
    {
      walktime = ares.walkDur.at(1) < MAX_WALK_TIME;
      walklength = areq.maxWalkDist >= ares.walkDur.at(1) * WALK_SPEED;
      timewindow = areq.duration >= (ares.dropoffTime - ares.pickupTime) + ares.walkDur.at(1); // evtl egal -> dann länge der kante anpassen ?
      //printf("checking: waiting: %d; walktime: %d; walklength: %d; timewindow: %d\n", waiting, walktime, walklength, timewindow);
      result = walklength && walktime && waiting && timewindow;
    }
    else
    {
      walktime = ares.walkDur.at(0) < MAX_WALK_TIME && ares.walkDur.at(1) < MAX_WALK_TIME
                 && areq.departureTime + ares.walkDur.at(0) < ares.pickupTime;
      walklength = areq.maxWalkDist >= ares.walkDur.at(1) * WALK_SPEED && areq.maxWalkDist >= ares.walkDur.at(0) * WALK_SPEED;
      timewindow = areq.duration >= (ares.dropoffTime - ares.pickupTime) + ares.walkDur.at(0) + ares.walkDur.at(1);
      //printf("checking: waiting: %d; walktime: %d; walklength: %d; timewindow: %d\n", waiting, walktime, walklength, timewindow);
      result = walklength && walktime && waiting && timewindow;
    }
  }
  return result;
}

struct server_info{
  std::string key_name;
  std::string header_first;
  std::string header_second;
  std::string first_addr;
  std::string second_addr;
  std::string id;
};

std::vector<server_info> get_server_info()
{
  std::vector<server_info> result;
  opt::variables_map var_map;
  opt::options_description description("Server");
  description.add_options()
      ("address", opt::value<std::string>()->required())
      ("address2", opt::value<std::string>())
      ("productid", opt::value<std::string>())
      ("hdr0", opt::value<std::string>())
      ("hdr1", opt::value<std::string>())
      ("hdr2", opt::value<std::string>())
      ("hdr3", opt::value<std::string>())
      ("hdr4", opt::value<std::string>())
      ("hdr5", opt::value<std::string>())
      ("hdr6", opt::value<std::string>())
      ("hdr7", opt::value<std::string>())
      ("hdr8", opt::value<std::string>());
  try {
    opt::store(opt::parse_config_file<char>("ondemand_server.cfg", description), var_map);
  } catch (const opt::reading_file& er) {
    LOG(logging::error) << " an error occured while reading ondemand_server.cfg file "
                        << er.what() << "!";
  }
  try {
    opt::notify(var_map);
  } catch (const opt::required_option& e) {
    LOG(logging::error) << " a required option is NOT set "
                        << e.what() << "!"
                        << "please check ondemand_server.cfg file";
  }

  for(auto it = var_map.begin(); it != var_map.end(); ++it)
  {
    server_info si;
    si.key_name = it->first;
    opt::variable_value value = it->second;
    std::string sval;
    if(!value.empty())
    {
      const type_info& type = value.value().type();
      if (type == typeid(std::string))
      {
        sval = value.as<std::string>();
      }
    }
    if(si.key_name == "address")
    {
      si.first_addr = sval;
    }
    else if(si.key_name == "address2")
    {
      si.second_addr = sval;
    }
    else if(si.key_name == "productid")
    {
      si.id = sval;
    }
    else
    {
      size_t idx = sval.find(',');
      si.header_first = sval.substr(0, idx);
      si.header_second = sval.substr(idx+1);
    }
    result.emplace_back(si);
  }
  return result;
}

availability_response check_od_availability(availability_request areq)
{
  //printf("check_od_availability!\n");

  std::vector<server_info> all_server_info = get_server_info();
  std::string addr;
  std::string addr2;
  std::map<std::string, std::string> hdrs;
  for(auto it = all_server_info.begin(); it != all_server_info.end(); ++it)
  {
    if(!it->header_first.empty() && !it->header_second.empty())
    {
      hdrs.insert(std::pair<std::string, std::string>(it->header_first,it->header_second));
    }
    else if(it->key_name == "address")
    {
      addr = it->first_addr;
    }
    else if(it->key_name == "address2")
    {
      addr2 = it->second_addr;
    }
    else if(it->key_name == "productid")
    {
      areq.productID = it->id;
    }
  }

  request::method m = request::GET;
  request req(addr, m, hdrs, "");

  geo::latlng req_dot_start;
  geo::latlng req_dot_end;
  req_dot_end.lat_ = areq.endpoint.lat_;
  req_dot_end.lng_ = areq.endpoint.lng_;
  req_dot_start.lat_ = areq.startpoint.lat_;
  req_dot_start.lng_ = areq.startpoint.lng_;
  std::vector<geo::latlng> req_dots;
  req_dots.emplace_back(req_dot_start);
  req_dots.emplace_back(req_dot_end);

  response firstresult = motis_http(req)->val();
  availability_response response_first = read_result(firstresult, true, req_dots);
  if(!response_first.available)
  {
    return response_first;
  }
  else
  {
    request::method m2 = request::POST;
    //UUID uuid;
    //UuidCreate(&uuid);
    //char* random_uuid_str;
    //UuidToStringA(&uuid, (RPC_CSTR*)&random_uuid_str);
    //hdrs.insert(pair<string, string>("Idempotency-Key", random_uuid_str));
    std::string body = create_json_body(areq);
    request req2(addr2, m2, hdrs, body);
    response secondresult = motis_http(req2)->val();
    availability_response response_second = read_result(secondresult, false, req_dots);
    response_second.available = checking(areq, response_second);
    return response_second;
  }
}

} // namespace intermodal