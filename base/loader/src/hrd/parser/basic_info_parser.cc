#include "motis/loader/hrd/parser/basic_info_parser.h"

#include "utl/parser/arg_parser.h"
#include "utl/verify.h"

#include "motis/core/common/date_time_util.h"
#include "motis/core/common/logging.h"

using namespace flatbuffers64;
using namespace utl;
using namespace motis::logging;

namespace motis::loader::hrd {

void verify_line_format(cstr s) {
  utl::verify(
      s.len == 10 &&
          (std::isdigit(s[0]) != 0 && std::isdigit(s[1]) != 0 && s[2] == '.' &&
           std::isdigit(s[3]) != 0 && std::isdigit(s[4]) != 0 && s[5] == '.' &&
           std::isdigit(s[6]) != 0 && std::isdigit(s[7]) != 0 &&
           std::isdigit(s[8]) != 0 && std::isdigit(s[9]) != 0),
      "interval boundary [{}] invalid", s.view());
}

std::tuple<std::uint16_t, std::uint16_t, std::uint16_t> yyyymmdd(
    cstr ddmmyyyy) {
  return std::make_tuple(parse<int>(ddmmyyyy.substr(6, size(4))),
                         parse<int>(ddmmyyyy.substr(3, size(2))),
                         parse<int>(ddmmyyyy.substr(0, size(2))));
}

unixtime str_to_unixtime(cstr s) {
  int year = 0, month = 0, day = 0;
  std::tie(year, month, day) = yyyymmdd(s);
  return to_unix_time(year, month, day);
}

std::pair<cstr, cstr> mask_dates(cstr str) {
  cstr from_line, to_line;

  from_line = get_line(str).substr(0, size(10));
  while (from_line.starts_with("%")) {
    skip_line(str);
    from_line = get_line(str).substr(0, size(10));
  }

  skip_line(str);

  to_line = get_line(str).substr(0, size(10));
  while (to_line.starts_with("%")) {
    skip_line(str);
    to_line = get_line(str).substr(0, size(10));
  }

  verify_line_format(from_line);
  verify_line_format(to_line);

  return std::make_pair(from_line, to_line);
}

Interval parse_interval(loaded_file const& basic_info_file) {
  scoped_timer timer("parsing schedule interval");
  cstr first_date;
  cstr last_date;
  std::tie(first_date, last_date) = mask_dates(basic_info_file.content());
  return {str_to_unixtime(first_date), str_to_unixtime(last_date)};
}

boost::gregorian::date get_first_schedule_date(loaded_file const& lf) {
  std::uint16_t year = 0, month = 0, day = 0;
  std::tie(year, month, day) = yyyymmdd(mask_dates(lf.content()).first);
  return {year, month, day};
}

std::string parse_schedule_name(loaded_file const& basic_info_file) {
  cstr str = basic_info_file.content();
  skip_line(str);  // from
  skip_line(str);  // to
  return get_line(str).to_str();  // schedule name
}

}  // namespace motis::loader::hrd
