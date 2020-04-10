#include "motis/loader/hrd/parser/timezones_parser.h"

#include "boost/date_time/gregorian/gregorian_types.hpp"

#include "utl/parser/arg_parser.h"

#include "motis/core/common/date_time_util.h"
#include "motis/loader/hrd/parser/basic_info_parser.h"
#include "motis/loader/util.h"

namespace motis::loader::hrd {

using namespace boost::gregorian;
using namespace utl;

int eva_number(cstr str) { return parse<int>(str); }

int distance_to_midnight(cstr hhmm) { return hhmm_to_min(parse<int>(hhmm)); }

int bitfield_idx(cstr ddmmyyyy, date const& first_schedule_date) {
  date season_begin_date(parse<int>(ddmmyyyy.substr(4, size(4))),
                         parse<int>(ddmmyyyy.substr(2, size(2))),
                         parse<int>(ddmmyyyy.substr(0, size(2))));
  return (season_begin_date - first_schedule_date).days();
}

timezones parse_timezones(loaded_file const& timezones_file,
                          loaded_file const& basic_data_file, config const& c) {
  auto const first_schedule_date = get_first_schedule_date(basic_data_file);

  timezones tz;
  for_each_line(timezones_file.content(), [&](cstr line) {
    if (line.length() == 15) {
      auto first_valid_eva_number =
          eva_number(line.substr(c.tz_.type1_first_valid_eva_));
      auto it = tz.eva_to_tze_.find(first_valid_eva_number);
      utl::verify(it != end(tz.eva_to_tze_),
                  "missing timezone information for eva number: {}",
                  first_valid_eva_number);

      tz.eva_to_tze_[eva_number(line.substr(c.tz_.type1_eva_))] = it->second;
      return;
    }

    if ((isdigit(line[0]) != 0) && line.length() >= 47) {
      boost::optional<season_entry> opt_season_entry;
      if (!line.substr(14, size(33)).trim().empty()) {
        opt_season_entry.emplace(
            distance_to_midnight(line.substr(c.tz_.type3_dst_to_midnight1_)),
            bitfield_idx(line.substr(c.tz_.type3_bitfield_idx1_),
                         first_schedule_date),
            bitfield_idx(line.substr(c.tz_.type3_bitfield_idx2_),
                         first_schedule_date),
            distance_to_midnight(line.substr(c.tz_.type3_dst_to_midnight2_)),
            distance_to_midnight(line.substr(c.tz_.type3_dst_to_midnight3_)));
      }

      tz.timezone_entries_.push_back(std::make_unique<timezone_entry>(
          distance_to_midnight(line.substr(c.tz_.type2_dst_to_midnight_)),
          opt_season_entry));

      tz.eva_to_tze_[eva_number(line.substr(c.tz_.type2_eva_))] =
          tz.timezone_entries_.back().get();
    }
  });

  return tz;
}

}  // namespace motis::loader::hrd
