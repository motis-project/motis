#include "motis/loader/gtfs/calendar_date.h"

#include "utl/enumerate.h"
#include "utl/parser/csv.h"
#include "utl/progress_tracker.h"

#include "motis/core/common/logging.h"
#include "motis/loader/util.h"

using namespace utl;
using std::get;

namespace motis::loader::gtfs {

using gtfs_calendar_date = std::tuple<cstr, int, int>;
enum { service_id, date_column, exception_type };

static const column_mapping<gtfs_calendar_date> calendar_columns = {
    {"service_id", "date", "exception_type"}};

calendar_date read_date(gtfs_calendar_date const& gtfs_date) {
  calendar_date d;
  d.day_ = {static_cast<uint16_t>(yyyymmdd_year(get<date_column>(gtfs_date))),
            static_cast<uint16_t>(yyyymmdd_month(get<date_column>(gtfs_date))),
            static_cast<uint16_t>(yyyymmdd_day(get<date_column>(gtfs_date)))};
  d.type_ = get<exception_type>(gtfs_date) == 1 ? calendar_date::ADD
                                                : calendar_date::REMOVE;
  return d;
}

std::map<std::string, std::vector<calendar_date>> read_calendar_date(
    loaded_file f) {
  motis::logging::scoped_timer timer{"calendar dates"};
  std::map<std::string, std::vector<calendar_date>> services;
  auto const entries = read<gtfs_calendar_date>(f.content(), calendar_columns);
  auto progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Parse Calendar Dates")
      .out_bounds(0.F, 5.F)
      .in_high(entries.size());
  for (auto const& d : entries) {
    progress_tracker->increment();
    services[get<service_id>(d).to_str()].push_back(read_date(d));
  }
  return services;
}

}  // namespace motis::loader::gtfs
