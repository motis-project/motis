#pragma once

#include "geo/latlng.h"

#include "utl/to_vec.h"

#include "osr/lookup.h"
#include "osr/ways.h"

#include "boost/json.hpp"

namespace icc {

enum class status : bool { kActive, kInactive };

struct elevator {
  std::uint64_t id_;
  geo::latlng pos_;
  status status_;
};

std::vector<elevator> parse(std::string_view s) {
  return utl::to_vec(
      boost::json::parse(s).as_array(), [](boost::json::value const& e) {
        auto const& o = e.as_object();
        return elevator{
            e.at("equipmentnumber").as_uint64(),
            {e.at("geocoordY").as_double(), e.at("geocoordX").as_double()},
            e.at("state") == "ACTIVE" ? status::kActive : status::kInactive};
      });
}

}  // namespace icc