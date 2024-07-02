#include "icc/parse_location.h"

#include "boost/phoenix/core/reference.hpp"
#include "boost/spirit/include/qi.hpp"

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

}  // namespace icc