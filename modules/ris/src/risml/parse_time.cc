#include "motis/ris/risml/parse_time.h"

#include "boost/date_time/c_local_time_adjustor.hpp"
#include "boost/date_time/gregorian/gregorian_types.hpp"
#include "boost/date_time/local_time_adjustor.hpp"
#include "boost/date_time/local_timezone_defs.hpp"
#include "boost/date_time/posix_time/posix_time_types.hpp"

#include "utl/parser/arg_parser.h"

#include "motis/core/common/date_time_util.h"
#include "motis/core/common/unixtime.h"
#include "motis/ris/risml/common.h"

using namespace utl;
using namespace boost::posix_time;
using namespace boost::gregorian;

using date_t = ptime::date_type;
using dur_t = ptime::time_duration_type;

using dst_traits = boost::date_time::eu_dst_trait<date_t>;
using engine = boost::date_time::dst_calc_engine<date_t, dur_t, dst_traits>;
using adjustor = boost::date_time::local_adjustor<ptime, 1, engine>;

namespace motis::ris::risml {

inline unixtime to_unix_time(boost::posix_time::ptime const& t) {
  boost::posix_time::ptime epoch(boost::gregorian::date(1970, 1, 1));
  return (t - epoch).total_seconds();
}

unixtime parse_time(cstr const& raw) {
  // format YYYYMMDDhhmmssfff (fff = millis)
  if (raw.length() < 14) {
    throw std::runtime_error("bad time format (length < 14)");
  }

  date d(parse<int>(raw.substr(0, size(4))),  //
         parse<int>(raw.substr(4, size(2))),
         parse<int>(raw.substr(6, size(2))));

  time_duration t(parse<int>(raw.substr(8, size(2))),
                  parse<int>(raw.substr(10, size(2))),
                  parse<int>(raw.substr(12, size(2))));

  ptime local_time(d, t);
  return to_unix_time(adjustor::local_to_utc(local_time));
}

unixtime parse_schedule_time(context& ctx, cstr const& raw) {
  auto t = parse_time(raw);
  ctx.earliest_ = std::min(ctx.earliest_, t);
  ctx.latest_ = std::max(ctx.latest_, t);
  return t;
}

}  // namespace motis::ris::risml
