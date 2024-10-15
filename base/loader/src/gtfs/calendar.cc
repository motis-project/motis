#include "motis/loader/gtfs/calendar.h"

#include "utl/parser/csv.h"

#include "motis/loader/util.h"

using namespace utl;
using std::get;

namespace motis::loader::gtfs {

using gtfs_calendar =
    std::tuple<cstr, int, int, int, int, int, int, int, int, int>;
enum {
  service_id,
  monday,
  tuesday,
  wednesday,
  thursday,
  friday,
  saturday,
  sunday,
  start_date,
  end_date
};

static const column_mapping<gtfs_calendar> calendar_columns = {
    {"service_id", "monday", "tuesday", "wednesday", "thursday", "friday",
     "saturday", "sunday", "start_date", "end_date"}};

std::bitset<7> traffic_week_days(gtfs_calendar const& c) {
  std::bitset<7> days;
  days.set(0, get<sunday>(c) == 1);
  days.set(1, get<monday>(c) == 1);
  days.set(2, get<tuesday>(c) == 1);
  days.set(3, get<wednesday>(c) == 1);
  days.set(4, get<thursday>(c) == 1);
  days.set(5, get<friday>(c) == 1);
  days.set(6, get<saturday>(c) == 1);
  return days;
}

std::map<std::string, calendar> read_calendar(loaded_file file) {
  if (file.empty()) {
    return {};
  }

  std::map<std::string, calendar> services;
  for (auto const& c : read<gtfs_calendar>(file.content(), calendar_columns)) {
    services.insert(std::make_pair(
        get<service_id>(c).to_str(),
        calendar{traffic_week_days(c),
                 {static_cast<uint16_t>(yyyymmdd_year(get<start_date>(c))),
                  static_cast<uint16_t>(yyyymmdd_month(get<start_date>(c))),
                  static_cast<uint16_t>(yyyymmdd_day(get<start_date>(c)))},
                 {static_cast<uint16_t>(yyyymmdd_year(get<end_date>(c))),
                  static_cast<uint16_t>(yyyymmdd_month(get<end_date>(c))),
                  static_cast<uint16_t>(yyyymmdd_day(get<end_date>(c)))}}));
  }
  return services;
}

}  // namespace motis::loader::gtfs
