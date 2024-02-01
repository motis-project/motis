#include "motis/paxmon/eval/forecast/pax_check_data.h"

#include <cstdint>
#include <fstream>

#include "utl/parser/buf_reader.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/file.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/for_each.h"
#include "utl/verify.h"

#include "motis/core/common/logging.h"
#include "motis/core/access/station_access.h"

#include "motis/paxmon/util/get_station_idx.h"

using namespace motis::paxmon::util;

namespace motis::paxmon::eval::forecast {

struct csv_row {
  utl::csv_col<std::uint64_t, UTL_NAME("ref")> ref_;

  utl::csv_col<utl::cstr, UTL_NAME("order_id")> order_id_;
  utl::csv_col<utl::cstr, UTL_NAME("trip_id")> trip_id_;

  utl::csv_col<utl::cstr, UTL_NAME("min_date")> min_date_;
  utl::csv_col<utl::cstr, UTL_NAME("max_date")> max_date_;

  utl::csv_col<utl::cstr, UTL_NAME("category")> category_;
  utl::csv_col<std::uint32_t, UTL_NAME("train_nr")> train_nr_;

  // 0: not checked
  // 1: ticket checked
  // 2: checkin
  // 3: both
  utl::csv_col<std::uint8_t, UTL_NAME("check_type")> check_type_;
  utl::csv_col<std::uint8_t, UTL_NAME("check_count")> check_count_;

  // 0: not checked, covered by other checked leg
  // 1: checked, plan = is
  // 2: checked, plan != is, matching leg found (exact match)
  // 3: checked, plan != is, matching leg found (equivalent stations)
  // 4: checked, plan != is, no matching leg found
  // 5: not checked, not covered by other leg
  utl::csv_col<std::uint8_t, UTL_NAME("leg_status")> leg_status_;

  utl::csv_col<std::uint8_t, UTL_NAME("planned_train")> planned_train_;
  utl::csv_col<std::uint8_t, UTL_NAME("checked_in_train")> checked_in_train_;
  utl::csv_col<std::uint8_t, UTL_NAME("canceled")> canceled_;

  utl::csv_col<utl::cstr, UTL_NAME("leg_start_id")> leg_start_id_;
  utl::csv_col<utl::cstr, UTL_NAME("leg_destination_id")> leg_destination_id_;

  utl::csv_col<std::time_t, UTL_NAME("leg_start_time")> leg_start_time_;
  utl::csv_col<std::time_t, UTL_NAME("leg_destination_time")>
      leg_destination_time_;

  utl::csv_col<utl::cstr, UTL_NAME("checkin_start_id")> checkin_start_id_;
  utl::csv_col<utl::cstr, UTL_NAME("checkin_destination_id")>
      checkin_destination_id_;

  utl::csv_col<std::time_t, UTL_NAME("check_min_time")> check_min_time_;
  utl::csv_col<std::time_t, UTL_NAME("check_max_time")> check_max_time_;

  utl::csv_col<std::time_t, UTL_NAME("schedule_train_start")>
      schedule_train_start_time_;

  // 0: unknown
  // 1: outward
  // 2: return
  utl::csv_col<std::uint8_t, UTL_NAME("direction")> direction_;

  utl::csv_col<std::uint64_t, UTL_NAME("planned_trip_ref")> planned_trip_ref_;
};

void load_pax_check_data(schedule const& sched, std::string const& filename,
                         pax_check_data& data) {
  data.clear();
  auto buf = utl::file(filename.data(), "r").content();
  auto file_content = utl::cstr{buf.data(), buf.size()};
  auto has_schedule_train_start_time = 0ULL;
  utl::line_range{utl::buf_reader{file_content}}  //
      | utl::csv<csv_row>()  //
      |
      utl::for_each([&](csv_row const& row) {
        auto const key = train_pax_data_key{
            mcd::string{row.category_.val().view()}, row.train_nr_.val()};
        auto& train_data = data.trains_[key];
        if (row.schedule_train_start_time_.val() != 0) {
          ++has_schedule_train_start_time;
        }
        train_data.entries_.emplace_back(pax_check_entry{
            .ref_ = row.ref_.val(),
            .order_id_ = mcd::string{row.order_id_.val().view()},
            .trip_id_ = mcd::string{row.trip_id_.val().view()},
            .check_type_ = static_cast<check_type>(row.check_type_.val()),
            .check_count_ = row.check_count_.val(),
            .leg_status_ = static_cast<leg_status>(row.leg_status_.val()),
            .direction_ = static_cast<travel_direction>(row.direction_.val()),
            .planned_train_ = row.planned_train_.val() != 0,
            .checked_in_train_ = row.checked_in_train_.val() != 0,
            .canceled_ = row.canceled_.val() != 0,
            .leg_start_station_ =
                get_station_idx(sched, row.leg_start_id_.val().view())
                    .value_or(0),
            .leg_destination_station_ =
                get_station_idx(sched, row.leg_destination_id_.val().view())
                    .value_or(0),
            .leg_start_time_ = unix_to_motistime(sched.schedule_begin_,
                                                 row.leg_start_time_.val()),
            .leg_destination_time_ = unix_to_motistime(
                sched.schedule_begin_, row.leg_destination_time_.val()),
            .checkin_start_station_ =
                get_station_idx(sched, row.checkin_start_id_.val().view())
                    .value_or(0),
            .checkin_destination_station_ =
                get_station_idx(sched, row.checkin_destination_id_.val().view())
                    .value_or(0),
            .check_min_time_ = unix_to_motistime(sched.schedule_begin_,
                                                 row.check_min_time_.val()),
            .check_max_time_ = unix_to_motistime(sched.schedule_begin_,
                                                 row.check_max_time_.val()),
            .schedule_train_start_time_ = unix_to_motistime(
                sched.schedule_begin_, row.schedule_train_start_time_.val()),
            .category_ = key.category_,
            .train_nr_ = key.train_nr_,
            .planned_trip_ref_ = row.planned_trip_ref_.val()});
      });

  for (auto const& train_entry : data.trains_) {
    for (auto const& entry : train_entry.second.entries_) {
      data.entries_by_order_id_[entry.order_id_].emplace_back(&entry);
    }
  }
}

}  // namespace motis::paxmon::eval::forecast
