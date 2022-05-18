#include "motis/paxmon/vubasisformation_parser.h"

#include <algorithm>
#include <iterator>

#include "rapidjson/document.h"
#include "rapidjson/error/en.h"

#include "motis/core/common/logging.h"

#include "motis/json/json.h"

#include "utl/to_vec.h"

using namespace motis::logging;
using namespace motis::json;

namespace motis::paxmon {

mcd::string get_eva(rapidjson::Value const& parent, char const* stop_key) {
  auto const& stop = get_obj(parent, stop_key);
  return get_str(stop, "evanummer");
}

std::vector<std::uint64_t> parse_vehicles(rapidjson::Value const& grp) {
  auto const& vehicles = get_array(grp, "allFahrzeug");
  return utl::to_vec(vehicles, [&](auto const& vehicle) {
    return get_parsed_number<std::uint64_t>(vehicle, "fahrzeugnummer");
  });
}

void parse_vubasisformation(std::string_view msg, capacity_maps& caps) {
  rapidjson::Document doc;
  if (doc.Parse(msg.data(), msg.size()).HasParseError()) {
    doc.GetParseError();
    LOG(error) << "VuBasisFormation: Bad JSON: "
               << rapidjson::GetParseError_En(doc.GetParseError())
               << " at offset " << doc.GetErrorOffset();
    return;
  }
  try {
    utl::verify(doc.IsObject(), "no root object");
    auto const& data = get_obj(doc, "data");
    auto const trip_uuid = get_uuid(data, "fahrtid");

    // reset existing data
    auto& vo = caps.trip_vehicle_map_[trip_uuid];
    vo.station_ranges_.clear();
    vo.sections_.clear();

    auto const sections_data = get_array(data, "allFahrtabschnitt");
    for (auto const& sec : sections_data) {
      auto const& dep = get_obj(sec, "abfahrt");
      auto const& groups = get_array(dep, "allFahrzeuggruppe");
      auto& sec_vo = vo.sections_.emplace_back();
      sec_vo.departure_eva_ = get_eva(dep, "haltestelle");
      sec_vo.departure_uuid_ = get_uuid(dep, "abfahrtid");

      for (auto const& grp : groups) {
        auto const from_eva = get_eva(grp, "starthaltestelle");
        auto const to_eva = get_eva(grp, "zielhaltestelle");
        auto const uics = parse_vehicles(grp);
        std::copy(begin(uics), end(uics),
                  std::back_inserter(sec_vo.vehicles_.uics_));
        auto& range_vehicles =
            vo.station_ranges_[station_range{from_eva, to_eva}].uics_;
        for (auto const& uic : uics) {
          if (std::find(begin(range_vehicles), end(range_vehicles), uic) !=
              end(range_vehicles)) {
            range_vehicles.emplace_back(uic);
          }
        }
      }
    }

  } catch (std::runtime_error const& e) {
    LOG(error) << "unable to parse VuBasisFormation message: " << e.what();
  }
}

}  // namespace motis::paxmon
