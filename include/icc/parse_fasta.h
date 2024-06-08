#pragma once

#include "geo/latlng.h"

#include "utl/enumerate.h"

#include "osr/lookup.h"
#include "osr/ways.h"

#include "boost/json.hpp"

namespace icc {

enum class status : bool { kActive, kInactive };

using elevator_idx_t = cista::strong<std::uint32_t, struct elevator_idx_>;

struct elevator {
  std::int64_t id_;
  geo::latlng pos_;
  status status_;
  std::string desc_;
};

nigiri::vector_map<elevator_idx_t, elevator> parse_fasta(std::string_view s) {
  auto ret = nigiri::vector_map<elevator_idx_t, elevator>{};
  for (auto const& [i, e] : utl::enumerate(boost::json::parse(s).as_array())) {
    if (e.at("type") != "ELEVATOR") {
      continue;
    }

    try {
      auto const& o = e.as_object();

      if (!o.contains("geocoordY") || !o.contains("geocoordX") ||
          !o.contains("state")) {
        std::cout << "skip: missing attributes: " << o << "\n";
        continue;
      }

      auto const id = o.contains("equipmentnumber")
                          ? e.at("equipmentnumber").to_number<std::int64_t>()
                          : 0U;
      ret.emplace_back(elevator{
          id,
          {e.at("geocoordY").to_number<double>(),
           e.at("geocoordX").to_number<double>()},
          e.at("state") == "ACTIVE" ? status::kActive : status::kInactive,
          o.contains("description")
              ? std::string{o.at("description").as_string()}
              : ""});
    } catch (std::exception const& ex) {
      std::cout << "error on value: " << e << ": " << ex.what() << "\n";
    }
  }
  return ret;
}

}  // namespace icc