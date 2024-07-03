#include "icc/parse_location.h"

#include "boost/phoenix/core/reference.hpp"
#include "boost/spirit/include/qi.hpp"

#include "date/date.h"

namespace n = nigiri;

namespace icc {

std::optional<osr::location> parse_location(std::string_view s) {
  using boost::phoenix::ref;
  using boost::spirit::ascii::space;
  using boost::spirit::qi::double_;
  using boost::spirit::qi::phrase_parse;

  auto first = begin(s);
  auto last = end(s);

  auto pos = geo::latlng{};
  auto level = 0.0F;
  auto const lat = [&](double& x) { pos.lat_ = x; };
  auto const lng = [&](double& x) { pos.lng_ = x; };
  auto const lvl = [&](double& x) { level = static_cast<float>(x); };

  auto const has_matched = phrase_parse(
      first, last,
      ((double_[lat] >> ',' >> double_[lng] >> ',' >> double_[lvl]) |
       double_[lat] >> ',' >> double_[lng]),
      space);
  if (!has_matched || first != last) {
    return std::nullopt;
  }

  return osr::location{pos, osr::to_level(level)};
}

n::unixtime_t get_date_time(std::optional<std::string> const& date,
                            std::optional<std::string> const& time) {
  if (!date.has_value()) {
    utl::verify(!time.has_value(), "time without date no supported");
    return std::chrono::time_point_cast<n::i32_minutes>(
        std::chrono::system_clock::now());
  } else {
    utl::verify(time.has_value(), "date without time not supported");
    auto const date_time = *date + " " + *time;

    // 06-28-2024 7:06pm
    // 06-28-2024 19:06
    std::stringstream ss;
    ss << date_time;

    auto t = n::unixtime_t{};
    if (date_time.contains("AM") || date_time.contains("PM")) {
      ss >> date::parse("%m-%d-%Y %I:%M %p", t);
    } else {
      ss >> date::parse("%m-%d-%Y %H:%M", t);
    }

    std::cout << "INPUT: " << date_time << "\n";
    std::cout << "OUTPUT: " << date::format("%m-%d-%Y %I:%M %p", t) << "\n";
    return t;
  }
}

}  // namespace icc