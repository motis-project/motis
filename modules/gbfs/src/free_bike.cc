#include "motis/gbfs/free_bike.h"

#include "rapidjson/document.h"
#include "rapidjson/error/en.h"

#include "utl/pipes.h"
#include "utl/verify.h"

#include "motis/core/common/logging.h"

#include "motis/json/json.h"

using namespace motis::logging;
using namespace motis::json;

namespace motis::gbfs {

std::vector<free_bike> parse_free_bikes(std::string const& tag,
                                        std::string_view s) {
  rapidjson::Document doc;
  if (doc.Parse(s.data(), s.size()).HasParseError()) {
    doc.GetParseError();
    throw utl::fail("GBFS free_bikes: Bad JSON: {} at offset {}, json={}",
                    rapidjson::GetParseError_En(doc.GetParseError()),
                    doc.GetErrorOffset(), s);
  }

  auto const& stations = get_array(get_obj(doc, "data"), "bikes");
  auto i = 0;
  return utl::all(stations)  //
         | utl::remove_if([&i](rapidjson::Value const& b) {
             auto const is_filtered =
                 !b.HasMember("lat") || !get_value(b, "lat").IsDouble() ||
                 !b.HasMember("lon") || !get_value(b, "lon").IsDouble();
             if (is_filtered) {
               l(warn, "GBFS station {} without position", i);
             }
             ++i;
             return is_filtered;
           })  //
         | utl::transform([&](rapidjson::Value const& s) {
             auto const pos = geo::latlng{get_value(s, "lat").GetDouble(),
                                          get_value(s, "lon").GetDouble()};

             auto type = std::string{};
             if (auto const it = s.FindMember("vehicle_type_id");
                 it != s.MemberEnd() && it->value.IsString()) {
               type = std::string{it->value.GetString(),
                                  it->value.GetStringLength()};
             }

             auto id = std::string{};
             if (auto const bike_id = s.FindMember("bike_id");
                 bike_id != s.MemberEnd() && bike_id->value.IsString()) {
               id = std::string{bike_id->value.GetString(),
                                bike_id->value.GetStringLength()};
             } else if (auto const vehicle_id = s.FindMember("vehicle_id");
                        vehicle_id != s.MemberEnd() &&
                        vehicle_id->value.IsString()) {
               id = std::string{vehicle_id->value.GetString(),
                                vehicle_id->value.GetStringLength()};
             } else {
               throw utl::fail("GBFS vehicle without bike_id/vehicle_id");
             }

             return free_bike{
                 .id_ = tag + id, .pos_ = pos, .type_ = std::move(type)};
           })  //
         | utl::vec();
}

}  // namespace motis::gbfs
