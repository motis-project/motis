#include "motis/elevators/parse_fasta.h"

#include <iostream>

#include "boost/json.hpp"

#include "date/date.h"

#include "utl/enumerate.h"

namespace n = nigiri;
namespace json = boost::json;

namespace motis {

n::unixtime_t parse_date_time(std::string_view s) {
  auto t = n::unixtime_t{};
  auto ss = std::stringstream{};
  ss.exceptions(std::ios_base::failbit | std::ios_base::badbit);
  ss << s;
  ss >> date::parse("%FT%T", t);
  return t;
}

std::vector<n::interval<n::unixtime_t>> parse_out_of_service(
    json::object const& o) {
  auto ret = std::vector<n::interval<n::unixtime_t>>{};

  if (!o.contains("outOfService")) {
    return ret;
  }

  for (auto const& entry : o.at("outOfService").as_array()) {
    auto const& interval = entry.as_array();
    if (interval.size() != 2U ||
        !(interval[0].is_string() && interval[1].is_string())) {
      fmt::println("skip: unable to parse out of service interval {}",
                   json::serialize(entry));
      continue;
    }
    ret.emplace_back(n::interval{parse_date_time(interval[0].as_string()),
                                 parse_date_time(interval[1].as_string())});
  }

  return ret;
}

std::optional<elevator> parse_elevator(json::value const& e) {
  if (e.at("type") != "ELEVATOR") {
    return std::nullopt;
  }

  try {
    auto const& o = e.as_object();

    if (!o.contains("geocoordY") || !o.contains("geocoordX") ||
        !o.contains("state")) {
      std::cout << "skip: missing attributes: " << o << "\n";
      return std::nullopt;
    }

    auto const id = o.contains("equipmentnumber")
                        ? e.at("equipmentnumber").to_number<std::int64_t>()
                        : 0U;
    return elevator{id,
                    std::nullopt,
                    geo::latlng{e.at("geocoordY").to_number<double>(),
                                e.at("geocoordX").to_number<double>()},
                    e.at("state").as_string() != "INACTIVE",
                    o.contains("description")
                        ? std::string{o.at("description").as_string()}
                        : "",
                    parse_out_of_service(o)};
  } catch (std::exception const& ex) {
    std::cout << "error on value: " << e << ": " << ex.what() << "\n";
    return std::nullopt;
  }
}

vector_map<elevator_idx_t, elevator> parse_fasta(std::string_view s) {
  auto ret = vector_map<elevator_idx_t, elevator>{};
  for (auto const [i, e] : utl::enumerate(json::parse(s).as_array())) {
    if (auto x = parse_elevator(e); x.has_value()) {
      ret.emplace_back(std::move(*x));
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

}  // namespace motis
