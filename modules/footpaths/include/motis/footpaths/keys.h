#pragma once

#include "geo/latlng.h"

#include "motis/footpaths/transfers.h"
#include "motis/footpaths/types.h"

namespace motis::footpaths {

inline key64_t to_key(geo::latlng const& coord) {
  auto latf = static_cast<float>(coord.lat_);
  auto lngf = static_cast<float>(coord.lng_);

  auto latb = (*reinterpret_cast<std::uint32_t*>(&latf));
  auto lngb = (*reinterpret_cast<std::uint32_t*>(&lngf));

  return key64_t{static_cast<std::uint64_t>(latb) << 32 | lngb};
}

inline string to_key(platform const& pf) {
  return {fmt::format("{}:{}", std::to_string(pf.info_.osm_id_),
                      get_osm_str_type(pf.info_.osm_type_))};
}

inline string to_key(transfer_request_keys const& treq_k) {
  return {fmt::format("{}{}", treq_k.from_nloc_key_, treq_k.profile_)};
}

inline string to_key(transfer_request const& treq) {
  return {fmt::format("{}{}", treq.from_nloc_key_, treq.profile_)};
}

inline string to_key(transfer_result const& tres) {
  return {fmt::format("{}{}", tres.from_nloc_key_, tres.profile_)};
}

}  // namespace motis::footpaths
