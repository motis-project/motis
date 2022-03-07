#pragma once

#include <string>
#include <vector>

#include "geo/latlng.h"

namespace motis::parking::parkendd {

enum class parkendd_state { NODATA, OPEN, CLOSED };

struct api_parking_lot {
  inline bool has_data() const { return state_ != parkendd_state::NODATA; }

  inline bool is_usable() const {
    return state_ == parkendd_state::OPEN && free_ > 0;
  }

  std::string id_;
  geo::latlng location_;
  int free_{};
  int total_{};
  parkendd_state state_{parkendd_state::NODATA};
};

std::vector<api_parking_lot> parse(std::string const& json);

}  // namespace motis::parking::parkendd
