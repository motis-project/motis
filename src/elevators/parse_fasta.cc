#include "icc/elevators/parse_fasta.h"

#include <iostream>

#include "boost/json.hpp"

#include "date/date.h"

#include "utl/enumerate.h"

namespace n = nigiri;
namespace json = boost::json;

namespace icc {

n::unixtime_t parse_date_time(std::string_view s) {
  auto t = n::unixtime_t{};
  auto ss = std::stringstream{};
  ss.exceptions(std::ios_base::failbit | std::ios_base::badbit);
  ss << s;
  ss >> date::parse("%FT%T", t);
  return t;
}

std::vector<n::interval<n::unixtime_t>> parse_out_of_service(
    json::array const& o) {
  auto ret = std::vector<n::interval<n::unixtime_t>>{};
  for (auto const& entry : o) {
    auto const& interval = entry.as_array();
    if (interval.size() != 2U ||
        !(interval[0].is_string() && interval[1].is_string())) {
      fmt::println("skip: unable to parse out of service interval {}",
                   json::serialize(entry));
      continue;
    }
    ret.emplace_back(parse_date_time(interval[0].as_string()),
                     parse_date_time(interval[1].as_string()));
  }
  return ret;
}

vector_map<elevator_idx_t, elevator> parse_fasta(std::string_view s) {
  auto ret = vector_map<elevator_idx_t, elevator>{};
  for (auto const& [i, e] : utl::enumerate(json::parse(s).as_array())) {
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
      ret.emplace_back(
          elevator{id,
                   {e.at("geocoordY").to_number<double>(),
                    e.at("geocoordX").to_number<double>()},
                   e.at("state").as_string() != "INACTIVE",
                   o.contains("description")
                       ? std::string{o.at("description").as_string()}
                       : "",
                   o.contains("outOfService")
                       ? parse_out_of_service(o.at("outOfService").as_array())
                       : std::vector<n::interval<n::unixtime_t>>{}});
    } catch (std::exception const& ex) {
      std::cout << "error on value: " << e << ": " << ex.what() << "\n";
    }
  }
  return ret;
}

vector_map<elevator_idx_t, elevator> parse_fasta(
    std::filesystem::path const& p) {
  return parse_fasta(
      cista::mmap{p.generic_string().c_str(), cista::mmap::protection::READ}
          .view());
}

}  // namespace icc