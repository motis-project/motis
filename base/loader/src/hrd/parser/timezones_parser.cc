#include "motis/loader/hrd/parser/timezones_parser.h"

#include "boost/date_time/gregorian/gregorian_types.hpp"

#include "utl/parser/arg_parser.h"

#include "motis/core/common/date_time_util.h"
#include "motis/loader/hrd/parser/basic_info_parser.h"
#include "motis/loader/util.h"

namespace motis::loader::hrd {

using namespace boost::gregorian;
using namespace utl;

int bitfield_idx(cstr ddmmyyyy,
                 boost::gregorian::date const& first_schedule_date) {
  boost::gregorian::date const season_begin_date(
      parse<int>(ddmmyyyy.substr(4, size(4))),
      parse<int>(ddmmyyyy.substr(2, size(2))),
      parse<int>(ddmmyyyy.substr(0, size(2))));
  return (season_begin_date - first_schedule_date).days();
}

int eva_number(cstr str) { return parse<int>(str); }

int distance_to_midnight(cstr hhmm) { return hhmm_to_min(parse<int>(hhmm)); }

std::vector<season_entry> parse_seasons(
    cstr const line, boost::gregorian::date const& schedule_begin) {
  enum state {
    kSeasonOffset,
    kSeasonBeginDate,
    kSeasonBeginHour,
    kSeasonEndDate,
    kSeasonEndHour,
    kNumStates
  } s{kSeasonOffset};
  std::vector<season_entry> seasons;
  auto e = season_entry{};
  for_each_token(line, ' ', [&](cstr const t) {
    switch (s) {
      case kSeasonOffset: e.gmt_offset_ = hhmm_to_min(parse<int>(t)); break;
      case kSeasonBeginDate:
        e.first_day_idx_ = bitfield_idx(t, schedule_begin);
        break;
      case kSeasonBeginHour:
        e.season_begin_time_ = hhmm_to_min(parse<int>(t));
        break;
      case kSeasonEndDate:
        e.last_day_idx_ = bitfield_idx(t, schedule_begin);
        break;
      case kSeasonEndHour:
        e.season_end_time_ = hhmm_to_min(parse<int>(t));
        seasons.push_back(e);
        break;
      default:;
    }
    s = static_cast<state>((s + 1) % kNumStates);
  });
  return seasons;
}

timezones parse_timezones(loaded_file const& timezones_file,
                          loaded_file const& basic_data_file, config const& c) {
  auto const first_schedule_date = get_first_schedule_date(basic_data_file);

  timezones tz;
  for_each_line(timezones_file.content(), [&](cstr line) {
    if (auto const comment_start = line.view().find('%');
        comment_start != std::string::npos) {
      line = line.substr(0, comment_start);
    }

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
      tz.timezone_entries_.emplace_back(std::make_unique<timezone_entry>(
          distance_to_midnight(line.substr(c.tz_.type2_dst_to_midnight_)),
          line.substr(14, size(33)).trim().empty()
              ? std::vector<season_entry>{}
              : parse_seasons(line.substr(14), first_schedule_date)));
      tz.eva_to_tze_[eva_number(line.substr(c.tz_.type2_eva_))] =
          tz.timezone_entries_.back().get();
    }
  });

  return tz;
}

}  // namespace motis::loader::hrd
