#include "motis/paxmon/capacity.h"

#include <cstdint>
#include <ctime>
#include <iterator>
#include <string>

#include "utl/parser/buf_reader.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/file.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/for_each.h"

#include "motis/core/common/logging.h"
#include "motis/core/access/station_access.h"

#include "motis/paxmon/util/get_station_idx.h"

using namespace motis::logging;
using namespace motis::paxmon::util;

namespace motis::paxmon {

namespace {

struct row {
  utl::csv_col<std::uint32_t, UTL_NAME("train_nr")> train_nr_;
  utl::csv_col<utl::cstr, UTL_NAME("category")> category_;
  utl::csv_col<utl::cstr, UTL_NAME("from")> from_;
  utl::csv_col<utl::cstr, UTL_NAME("to")> to_;
  utl::csv_col<std::time_t, UTL_NAME("departure")> departure_;
  utl::csv_col<std::time_t, UTL_NAME("arrival")> arrival_;
  utl::csv_col<std::uint16_t, UTL_NAME("seats")> seats_;
};

};  // namespace

std::size_t load_capacities(schedule const& sched,
                            std::string const& capacity_file,
                            trip_capacity_map_t& trip_map,
                            category_capacity_map_t& category_map) {
  auto buf = utl::file(capacity_file.data(), "r").content();
  auto const file_content = utl::cstr{buf.data(), buf.size()};
  auto entry_count = 0ULL;

  utl::line_range<utl::buf_reader>{file_content}  //
      | utl::csv<row>()  //
      | utl::for_each([&](auto&& row) {
          if (row.train_nr_ != 0) {
            auto const from_station_idx =
                get_station_idx(sched, row.from_.val().view()).value_or(0);
            auto const to_station_idx =
                get_station_idx(sched, row.to_.val().view()).value_or(0);
            time departure = row.departure_.val() != 0
                                 ? unix_to_motistime(sched.schedule_begin_,
                                                     row.departure_.val())
                                 : 0;
            time arrival = row.arrival_.val() != 0
                               ? unix_to_motistime(sched.schedule_begin_,
                                                   row.arrival_.val())
                               : 0;

            if (row.from_.val() && from_station_idx == 0) {
              LOG(warn) << "station not found: " << row.from_.val().view();
            }
            if (row.to_.val() && to_station_idx == 0) {
              LOG(warn) << "station not found: " << row.to_.val().view();
            }
            if (departure == INVALID_TIME || arrival == INVALID_TIME) {
              return;
            }

            auto const tid = cap_trip_id{row.train_nr_.val(), from_station_idx,
                                         to_station_idx, departure, arrival};
            trip_map[tid] = row.seats_.val();
            ++entry_count;
          } else if (row.category_.val()) {
            category_map[row.category_.val().view()] = row.seats_.val();
            ++entry_count;
          }
        });

  return entry_count;
}

std::pair<std::uint16_t, capacity_source> get_capacity(
    schedule const& sched, light_connection const& lc,
    trip_capacity_map_t const& trip_map,
    category_capacity_map_t const& category_map,
    std::uint16_t default_capacity) {
  auto seats = 0;

  for (auto const& trp : *sched.merged_trips_.at(lc.trips_)) {
    auto const train_nr = trp->id_.primary_.get_train_nr();
    auto const tid = cap_trip_id{train_nr, trp->id_.primary_.get_station_id(),
                                 trp->id_.secondary_.target_station_id_,
                                 trp->id_.primary_.get_time(),
                                 trp->id_.secondary_.target_time_};
    if (auto const lb = trip_map.lower_bound(tid); lb != end(trip_map)) {
      if (lb->first == tid) {
        return {lb->second, capacity_source::TRIP_EXACT};
      } else if (lb->first.train_nr_ == train_nr) {
        seats = lb->second;
      } else if (auto const prev = std::prev(lb);
                 prev != end(trip_map) && prev->first.train_nr_ == train_nr) {
        seats = prev->second;
      }
    }
  }

  if (seats != 0) {
    return {seats, capacity_source::TRAIN_NR};
  } else {
    auto const& category =
        sched.categories_[lc.full_con_->con_info_->family_]->name_;
    if (auto const it = category_map.find(category); it != end(category_map)) {
      return {it->second, capacity_source::CATEGORY};
    } else if (auto const it = category_map.find(std::to_string(
                   static_cast<service_class_t>(lc.full_con_->clasz_)));
               it != end(category_map)) {
      return {it->second, capacity_source::CLASZ};
    }
    return {default_capacity, capacity_source::DEFAULT};
  }
}

}  // namespace motis::paxmon
