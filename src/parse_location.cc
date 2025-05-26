#include "motis/parse_location.h"

#include <utility>

#include "boost/phoenix/core/reference.hpp"
#include "boost/spirit/include/qi.hpp"

#include "date/date.h"

#include "utl/parser/arg_parser.h"

namespace n = nigiri;

namespace motis {

std::optional<osr::location> parse_location(std::string_view s,
                                            char const separator) {
  using boost::phoenix::ref;
  using boost::spirit::ascii::space;
  using boost::spirit::qi::double_;
  using boost::spirit::qi::phrase_parse;

  auto first = begin(s);
  auto last = end(s);

  auto pos = geo::latlng{};
  auto level = osr::kNoLevel;
  auto const lat = [&](double& x) { pos.lat_ = x; };
  auto const lng = [&](double& x) { pos.lng_ = x; };
  auto const lvl = [&](double& x) {
    level = osr::level_t{static_cast<float>(x)};
  };

  auto const has_matched =
      phrase_parse(first, last,
                   ((double_[lat] >> separator >> double_[lng] >> separator >>
                     double_[lvl]) |
                    double_[lat] >> separator >> double_[lng]),
                   space);
  if (!has_matched || first != last) {
    return std::nullopt;
  }

  return osr::location{pos, level};
}

date::sys_days parse_iso_date(std::string_view s) {
  auto d = date::sys_days{};
  (std::stringstream{} << s) >> date::parse("%F", d);
  return d;
}

std::pair<n::direction, n::unixtime_t> parse_cursor(std::string_view s) {
  auto const split_pos = s.find("|");
  utl::verify(split_pos != std::string_view::npos && split_pos != s.size() - 1U,
              "invalid page cursor {}, separator '|' not found", s);

  auto const time_str = s.substr(split_pos + 1U);
  utl::verify(
      utl::all_of(time_str, [&](auto&& c) { return std::isdigit(c) != 0U; }),
      "invalid page cursor \"{}\", timestamp not a number", s);

  auto const t = n::unixtime_t{std::chrono::duration_cast<n::i32_minutes>(
      std::chrono::seconds{utl::parse<std::int64_t>(time_str)})};

  auto const direction = s.substr(0, split_pos);
  switch (cista::hash(direction)) {
    case cista::hash("EARLIER"): return {n::direction::kBackward, t};
    case cista::hash("LATER"): return {n::direction::kForward, t};
    default: throw utl::fail("invalid cursor: \"{}\"", s);
  }
}

n::routing::query cursor_to_query(std::string_view s) {
  auto const [dir, t] = parse_cursor(s);
  switch (dir) {
    case n::direction::kBackward:
      return n::routing::query{
          .start_time_ =
              n::routing::start_time_t{n::interval{t - n::duration_t{120}, t}},
          .extend_interval_earlier_ = true,
          .extend_interval_later_ = false};

    case n::direction::kForward:
      return n::routing::query{
          .start_time_ =
              n::routing::start_time_t{n::interval{t, t + n::duration_t{120}}},
          .extend_interval_earlier_ = false,
          .extend_interval_later_ = true};
  }
  std::unreachable();
}

}  // namespace motis