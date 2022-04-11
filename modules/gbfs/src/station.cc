#include "motis/gbfs/station.h"

#include "rapidjson/document.h"
#include "rapidjson/error/en.h"

#include "utl/pipes.h"
#include "utl/verify.h"

#include "motis/json/json.h"

#include "motis/pair.h"

using namespace motis::json;

namespace motis::gbfs {

void add_status(std::string const& tag,
                mcd::hash_map<std::string, station>& stations,
                std::string_view s) {
  rapidjson::Document doc;
  if (doc.Parse(s.data(), s.size()).HasParseError()) {
    doc.GetParseError();
    throw utl::fail("GBFS station_status: Bad JSON: {} at offset {}, in={}",
                    rapidjson::GetParseError_En(doc.GetParseError()),
                    doc.GetErrorOffset(), s);
  }

  auto const& in = get_array(get_obj(doc, "data"), "stations");
  for (auto const& x : in) {
    auto& station = stations.at(tag + std::string{get_str(x, "station_id")});
    if (auto const it = x.FindMember("num_bikes_available");
        it != x.MemberEnd() && it->value.IsNumber()) {
      station.bikes_available_ = it->value.GetUint();
    }
  }
}

mcd::hash_map<std::string, station> parse_stations(std::string const& tag,
                                                   std::string_view info,
                                                   std::string_view status) {
  rapidjson::Document doc;
  if (doc.Parse(info.data(), info.size()).HasParseError()) {
    doc.GetParseError();
    throw utl::fail(
        "GBFS station_information: Bad JSON: {} at offset {}, in={}",
        rapidjson::GetParseError_En(doc.GetParseError()), doc.GetErrorOffset(),
        info);
  }

  auto const& stations = get_array(get_obj(doc, "data"), "stations");
  auto station_map =
      utl::all(stations)  //
      | utl::transform([&](rapidjson::Value const& s) {
          auto const id = tag + std::string{get_str(s, "station_id")};
          auto const& lat = get_value(s, "lat");
          auto const& lon = get_value(s, "lon");
          utl::verify(lat.IsDouble() && lon.IsDouble(),
                      "station {} lat/lon err", id);
          auto const pos = geo::latlng{get_value(s, "lat").GetDouble(),
                                       get_value(s, "lon").GetDouble()};
          return mcd::pair{id, station{.id_ = id,
                                       .name_ = std::string{get_str(s, "name")},
                                       .pos_ = pos}};
        })  //
      | utl::emplace<mcd::hash_map<std::string, station>>();
  add_status(tag, station_map, status);
  return station_map;
}

}  // namespace motis::gbfs
