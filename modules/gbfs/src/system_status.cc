#include "motis/gbfs/system_status.h"

#include "rapidjson/document.h"
#include "rapidjson/error/en.h"

#include "cista/hash.h"

#include "utl/to_vec.h"

#include "motis/json/json.h"

using namespace motis::json;

namespace motis::gbfs {

std::vector<urls> read_system_status(std::string_view s) {
  rapidjson::Document doc;
  if (doc.Parse(s.data(), s.size()).HasParseError()) {
    doc.GetParseError();
    throw utl::fail("GBFS station_information: Bad JSON: {} at offset {}",
                    rapidjson::GetParseError_En(doc.GetParseError()),
                    doc.GetErrorOffset());
  }

  auto const& data = get_obj(doc, "data");
  return utl::to_vec(data.GetObject(), [](auto const& m) {
    auto u = urls{};
    u.lang_ = std::string{m.name.GetString(), m.name.GetStringLength()};
    for (auto const& f : get_array(m.value, "feeds")) {
      auto const name = get_str(f, "name");
      switch (cista::hash(name)) {
        case cista::hash("station_information"):
          u.station_info_url_ = get_str(f, "url");
          break;

        case cista::hash("station_status"):
          u.station_status_url_ = get_str(f, "url");
          break;

        case cista::hash("free_bike_status"):
        case cista::hash("vehicle_status"):
          u.free_bike_url_ = get_str(f, "url");
          break;
      }
    }
    return u;
  });
}

}  // namespace motis::gbfs
