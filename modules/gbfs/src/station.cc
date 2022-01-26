#include "motis/gbfs/station.h"

#include "rapidjson/document.h"
#include "rapidjson/error/en.h"

#include "motis/core/common/logging.h"

#include "utl/pipes.h"
#include "utl/verify.h"

#include "motis/json/json.h"

using namespace motis::logging;
using namespace motis::json;

namespace motis::gbfs {

std::vector<station> parse_stations(std::string_view s) {
  rapidjson::Document doc;
  if (doc.Parse(s.data(), s.size()).HasParseError()) {
    doc.GetParseError();
    throw utl::fail("GBFS station_information: Bad JSON: {} at offset {}",
                    rapidjson::GetParseError_En(doc.GetParseError()),
                    doc.GetErrorOffset());
  }

  auto const& stations = get_array(get_obj(doc, "data"), "stations");
  auto i = 0;
  return utl::all(stations)  //
         | utl::remove_if([&i](rapidjson::Value const& s) {
             auto const is_filtered =
                 !s.HasMember("lat") || !get_value(s, "lat").IsDouble() ||
                 !s.HasMember("lon") || !get_value(s, "lon").IsDouble() ||
                 !s.HasMember("station_id") ||
                 !get_value(s, "station_id").IsString() ||
                 !s.HasMember("name") || !get_value(s, "name").IsString();
             if (is_filtered) {
               l(warn, "GBFS station {} has missing values", i);
             }
             ++i;
             return is_filtered;
           })  //
         | utl::transform([](rapidjson::Value const& s) {
             auto const pos = geo::latlng{get_value(s, "lat").GetDouble(),
                                          get_value(s, "lon").GetDouble()};
             return station{.id_ = std::string{get_str(s, "station_id")},
                            .name_ = std::string{get_str(s, "name")},
                            .pos_ = pos};
           })  //
         | utl::vec();
}

}  // namespace motis::gbfs
