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

std::vector<free_bike> parse_free_bikes(std::string_view s) {
  rapidjson::Document doc;
  if (doc.Parse(s.data(), s.size()).HasParseError()) {
    doc.GetParseError();
    throw utl::fail("GBFS station_information: Bad JSON: {} at offset {}",
                    rapidjson::GetParseError_En(doc.GetParseError()),
                    doc.GetErrorOffset());
  }

  auto const& stations = get_array(get_obj(doc, "data"), "bikes");
  auto i = 0;
  return utl::all(stations)  //
         |
         utl::remove_if([](rapidjson::Value const& b) {
           auto const m = b.FindMember("is_reserved");
           return m != b.MemberEnd() && m->value.IsBool() && m->value.GetBool();
         }) |
         utl::remove_if([](rapidjson::Value const& b) {
           auto const m = b.FindMember("is_disabled");
           return m != b.MemberEnd() && m->value.IsBool() && m->value.GetBool();
         })  //
         | utl::remove_if([&i](rapidjson::Value const& b) {
             auto const is_filtered =
                 !b.HasMember("lat") || !get_value(b, "lat").IsDouble() ||
                 !b.HasMember("lon") || !get_value(b, "lon").IsDouble() ||
                 !b.HasMember("bike_id") || !get_value(b, "bike_id").IsString();
             if (is_filtered) {
               l(warn, "GBFS station {} has missing values", i);
             }
             ++i;
             return is_filtered;
           })  //
         |
         utl::transform([](rapidjson::Value const& s) {
           auto const pos = geo::latlng{get_value(s, "lat").GetDouble(),
                                        get_value(s, "lon").GetDouble()};
           auto const type = s.HasMember("vehicle_type_id") &&
                                     get_value(s, "vehicle_type_id").IsString()
                                 ? get_str(s, "vehicle_type_id")
                                 : std::string_view{""};
           return free_bike{.id_ = std::string{get_str(s, "bike_id")},
                            .pos_ = pos,
                            .type_ = std::string{type}};
         })  //
         | utl::vec();
}

}  // namespace motis::gbfs
