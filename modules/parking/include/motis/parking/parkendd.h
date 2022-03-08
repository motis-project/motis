#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "geo/latlng.h"

#include "motis/parking/parking_lot.h"

namespace motis::parking::parkendd {

enum class parkendd_state : std::uint8_t { NODATA, OPEN, CLOSED };

struct api_parking_lot {
  inline bool has_data() const { return state_ != parkendd_state::NODATA; }

  inline bool is_usable() const {
    return state_ == parkendd_state::OPEN && free_ > 0;
  }

  std::string id_;
  std::string name_;
  std::string lot_type_;
  std::string address_;
  geo::latlng location_;
  std::int32_t free_{};
  std::int32_t total_{};
  parkendd_state state_{parkendd_state::NODATA};
  std::int32_t parking_id_{};
};

std::vector<api_parking_lot> parse(std::string const& json);

}  // namespace motis::parking::parkendd
