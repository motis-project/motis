#include "motis/parking/parkendd.h"

#include "rapidjson/document.h"
#include "rapidjson/error/en.h"

#include "utl/to_vec.h"
#include "utl/verify.h"

#include "motis/json/json.h"

using namespace motis::json;

namespace motis::parking::parkendd {

geo::latlng parse_coords(rapidjson::Value const& coords) {
  return geo::latlng{get_double(coords, "lat"), get_double(coords, "lng")};
}

parkendd_state parse_state(std::string_view const str) {
  if (str == "open") {
    return parkendd_state::OPEN;
  } else if (str == "closed") {
    return parkendd_state::CLOSED;
  } else if (str == "nodata") {
    return parkendd_state::NODATA;
  } else {
    throw utl::fail("unsupported parking lot state: \"{}\"", str);
  }
}

api_parking_lot parse_lot(rapidjson::Value const& lot) {
  return api_parking_lot{std::string{get_str(lot, "id")},
                         std::string{get_str(lot, "name")},
                         std::string{get_str(lot, "lot_type")},
                         std::string{get_str(lot, "address")},
                         parse_coords(get_obj(lot, "coords")),
                         get_int(lot, "free"),
                         get_int(lot, "total"),
                         parse_state(get_str(lot, "state")),
                         0};
}

std::vector<api_parking_lot> parse(std::string const& json) {
  rapidjson::Document doc;
  doc.Parse(json.data(), json.size());
  utl::verify(!doc.HasParseError(), "bad json: {} at offset {}",
              rapidjson::GetParseError_En(doc.GetParseError()),
              doc.GetErrorOffset());

  utl::verify(doc.IsObject(), "no root object");
  auto const& lots = get_array(doc, "lots");
  return utl::to_vec(lots, [](auto const& lot) { return parse_lot(lot); });
}

}  // namespace motis::parking::parkendd
