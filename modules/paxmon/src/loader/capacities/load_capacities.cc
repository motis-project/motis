#include "motis/paxmon/loader/capacities/load_capacities.h"

#include <cstdint>
#include <ctime>
#include <algorithm>
#include <fstream>
#include <functional>
#include <iterator>
#include <set>
#include <string>
#include <string_view>
#include <utility>

#include "utl/parser/buf_reader.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/file.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/for_each.h"
#include "utl/verify.h"

#include "motis/core/common/logging.h"
#include "motis/core/access/station_access.h"

#include "motis/paxmon/util/get_station_idx.h"

using namespace motis::logging;
using namespace motis::paxmon::util;

namespace motis::paxmon::loader::capacities {

struct trip_row {
  utl::csv_col<std::uint32_t, UTL_NAME("train_nr")> train_nr_;
  utl::csv_col<utl::cstr, UTL_NAME("category")> category_;
  utl::csv_col<utl::cstr, UTL_NAME("from")> from_;
  utl::csv_col<utl::cstr, UTL_NAME("from_name")> from_name_;
  utl::csv_col<utl::cstr, UTL_NAME("to")> to_;
  utl::csv_col<utl::cstr, UTL_NAME("to_name")> to_name_;
  utl::csv_col<std::time_t, UTL_NAME("departure")> departure_;
  utl::csv_col<std::time_t, UTL_NAME("arrival")> arrival_;
  utl::csv_col<std::uint16_t, UTL_NAME("seats")> seats_;
};

struct vehicle_row {
  utl::csv_col<std::uint64_t, UTL_NAME("uic_number")> uic_number_;
  utl::csv_col<utl::cstr, UTL_NAME("attribute_name")> attribute_name_;
  utl::csv_col<std::uint16_t, UTL_NAME("attribute_value")> attribute_value_;
};

csv_format get_csv_format(std::string_view const file_content) {
  if (auto const nl = file_content.find('\n'); nl != std::string_view::npos) {
    auto const header = file_content.substr(0, nl);
    utl::verify(header.find(',') != std::string_view::npos,
                "paxmon: only ',' separator supported for capacity csv files");
    if (header.find("seats") != std::string_view::npos) {
      return csv_format::TRIP;
    } else if (header.find("uic_number") != std::string_view::npos) {
      return csv_format::VEHICLE;
    } else {
      throw utl::fail("paxmon: unsupported capacity csv input format");
    }
  }
  throw utl::fail("paxmon: empty capacity input file");
}

load_capacities_result load_capacities(schedule const& sched,
                                       capacity_maps& caps,
                                       std::string_view const data) {
  auto file_content = utl::cstr{data};
  if (file_content.starts_with("\xEF\xBB\xBF")) {
    // skip utf-8 byte order mark (otherwise the first column is ignored)
    file_content = file_content.substr(3);
  }
  auto res = load_capacities_result{};
  res.format_ = get_csv_format(file_content.view());

  if (res.format_ == csv_format::TRIP) {
    utl::line_range<utl::buf_reader>{file_content}  //
        | utl::csv<trip_row>()  //
        |
        utl::for_each([&](trip_row const& row) {
          if (row.train_nr_.val() != 0) {
            auto const from_station_idx =
                get_station_idx(sched, row.from_.val().view()).value_or(0);
            auto const to_station_idx =
                get_station_idx(sched, row.to_.val().view()).value_or(0);
            time const departure =
                row.departure_.val() != 0
                    ? unix_to_motistime(sched.schedule_begin_,
                                        row.departure_.val())
                    : 0;
            time const arrival = row.arrival_.val() != 0
                                     ? unix_to_motistime(sched.schedule_begin_,
                                                         row.arrival_.val())
                                     : 0;

            if (row.from_.val() && from_station_idx == 0) {
              res.stations_not_found_.insert(row.from_.val().to_str());
            }
            if (row.to_.val() && to_station_idx == 0) {
              res.stations_not_found_.insert(row.to_.val().to_str());
            }
            if (departure == INVALID_TIME || arrival == INVALID_TIME) {
              ++res.skipped_entry_count_;
              return;
            }

            auto const tid = cap_trip_id{row.train_nr_.val(), from_station_idx,
                                         departure, to_station_idx, arrival};
            caps.trip_capacity_map_[tid] = row.seats_.val();
            ++res.loaded_entry_count_;
          } else if (row.category_.val()) {
            caps.category_capacity_map_[row.category_.val().view()] =
                row.seats_.val();
            ++res.loaded_entry_count_;
          }
        });
  } else if (res.format_ == csv_format::VEHICLE) {
    utl::line_range<utl::buf_reader>{file_content}  //
        | utl::csv<vehicle_row>()  //
        | utl::for_each([&](vehicle_row const& row) {
            auto& cap = caps.vehicle_capacity_map_[row.uic_number_.val()];
            auto const& attr = row.attribute_name_.val();
            auto const val = row.attribute_value_.val();
            if (attr == "SITZPL_GESAMT") {
              cap.seats_ = val;
            } else if (attr == "ANZAHL_STEHPL") {
              cap.standing_ = val;
            } else if (attr == "PERS_ZUGELASSEN") {
              cap.total_limit_ = val;
            } else if (attr.starts_with("ANZ_SITZ_1KL")) {
              cap.seats_1st_ += val;
              cap.update_seats();
            } else if (attr.starts_with("ANZ_SITZ_2KL")) {
              cap.seats_2nd_ += val;
              cap.update_seats();
            }
            ++res.loaded_entry_count_;
          });
  }

  return res;
}

load_capacities_result load_capacities_from_file(
    schedule const& sched, capacity_maps& caps,
    std::string const& capacity_file, std::string const& match_log_file) {
  auto buf = utl::file(capacity_file.data(), "r").content();
  auto res = load_capacities(
      sched, caps,
      std::string_view{reinterpret_cast<char const*>(buf.data()), buf.size()});

  if (!res.stations_not_found_.empty()) {
    LOG(warn) << res.stations_not_found_.size() << " stations not found";
    if (!match_log_file.empty()) {
      std::ofstream ml{match_log_file};
      ml << "stations not found:\n";
      for (auto const& id : res.stations_not_found_) {
        ml << id << "\n";
      }
      LOG(warn) << "capacity match log report written to: " << match_log_file;
    }
  }

  return res;
}

}  // namespace motis::paxmon::loader::capacities
