#pragma once

#include <cstdint>

#include "cista/containers/string.h"
#include "cista/containers/variant.h"

#include "geo/latlng.h"

namespace motis::parking {

enum class osm_type : std::uint8_t { NODE, WAY, RELATION };
enum class fee_type : std::uint8_t { UNKNOWN, NO, YES };

struct osm_parking_lot_info {
  std::int64_t osm_id_{};
  osm_type osm_type_{};
  fee_type fee_{fee_type::UNKNOWN};
};

enum class parkendd_state : std::uint8_t { NODATA, OPEN, CLOSED };

struct parkendd_parking_lot_info {
  cista::offset::string id_;
  cista::offset::string name_;
  cista::offset::string lot_type_;
  cista::offset::string address_;
};

struct parking_lot {
  using info_t =
      cista::offset::variant<osm_parking_lot_info, parkendd_parking_lot_info>;

  fee_type get_fee_type() const;

  inline bool valid() const { return id_ != 0; }

  std::int32_t id_{};
  geo::latlng location_{};

  info_t info_;
};

}  // namespace motis::parking
