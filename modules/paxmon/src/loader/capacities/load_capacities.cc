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

struct ris_service_vehicles_row {
  utl::csv_col<std::uint64_t, UTL_NAME("uic_number")> uic_number_;
  utl::csv_col<utl::cstr, UTL_NAME("attribute_name")> attribute_name_;
  utl::csv_col<std::uint16_t, UTL_NAME("attribute_value")> attribute_value_;
};

struct fzg_kap_row {
  utl::csv_col<std::uint64_t, UTL_NAME("FzNr")> uic_number_;
  utl::csv_col<std::uint16_t, UTL_NAME("Summe_Sitzplätze")> seats_;
};

struct fzg_gruppe_row {
  utl::csv_col<utl::cstr, UTL_NAME("FzgGruppe")> group_name_;
  utl::csv_col<std::uint16_t, UTL_NAME("Summe_Sitzplätze")> seats_;
  utl::csv_col<std::uint16_t, UTL_NAME("Summe_Sitzplätze1Kl")> seats_1st_;
  utl::csv_col<std::uint16_t, UTL_NAME("Summe_Sitzplätze2Kl")> seats_2nd_;
  utl::csv_col<std::uint16_t, UTL_NAME("Kritisch")> critical_;
};

struct gattung_row {
  utl::csv_col<utl::cstr, UTL_NAME("Gattung")> gattung_;
  utl::csv_col<std::uint16_t, UTL_NAME("Summe_Sitzplätze")> seats_;
  utl::csv_col<std::uint16_t, UTL_NAME("Summe_Sitzplätze1Kl")> seats_1st_;
  utl::csv_col<std::uint16_t, UTL_NAME("Summe_Sitzplätze2Kl")> seats_2nd_;
};

struct baureihe_row {
  utl::csv_col<utl::cstr, UTL_NAME("vehicle_abr")> vehicle_abr_;
  utl::csv_col<std::uint16_t, UTL_NAME("seats")> seats_;
  utl::csv_col<std::uint16_t, UTL_NAME("seats_1st")> seats_1st_;
  utl::csv_col<std::uint16_t, UTL_NAME("seats_2nd")> seats_2nd_;
  utl::csv_col<std::uint16_t, UTL_NAME("standing")> standing_;
  utl::csv_col<std::uint16_t, UTL_NAME("total_limit")> total_limit_;
};

struct detected_csv_format {
  csv_format type_{};
  csv_separator separator_{};
};

detected_csv_format get_csv_format(std::string_view const file_content) {
  if (auto const nl = file_content.find('\n'); nl != std::string_view::npos) {
    auto const header = file_content.substr(0, nl);
    auto const comma_sep = header.find(',') != std::string_view::npos;
    auto const semicolon_sep = header.find(';') != std::string_view::npos;
    utl::verify(comma_sep || semicolon_sep,
                "paxmon: unsupported capacity file, no csv header detected");
    utl::verify(!(comma_sep && semicolon_sep),
                "paxmon: unsupported capacity file, can't determine separator");
    auto const sep =
        comma_sep ? csv_separator::COMMA : csv_separator::SEMICOLON;
    if (header.find("vehicle_abr") != std::string_view::npos) {
      return {csv_format::BAUREIHE, sep};
    } else if (header.find("seats") != std::string_view::npos) {
      return {csv_format::TRIP, sep};
    } else if (header.find("uic_number") != std::string_view::npos) {
      return {csv_format::RIS_SERVICE_VEHICLES, sep};
    } else if (header.find("FzNr") != std::string_view::npos) {
      return {csv_format::FZG_KAP, sep};
    } else if (header.find("FzgGruppe") != std::string_view::npos) {
      return {csv_format::FZG_GRUPPE, sep};
    } else if (header.find("Gattung") != std::string_view::npos) {
      return {csv_format::GATTUNG, sep};
    } else {
      throw utl::fail("paxmon: unsupported capacity csv input format");
    }
  }
  throw utl::fail("paxmon: empty capacity input file");
}

template <char Separator>
load_capacities_result load_capacities(schedule const& sched,
                                       capacity_maps& caps,
                                       utl::cstr const& file_content,
                                       csv_format const format) {
  auto res = load_capacities_result{};
  res.format_ = format;

  if (format == csv_format::TRIP) {
    utl::line_range{utl::buf_reader{file_content}}  //
        | utl::csv<trip_row, Separator>()  //
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
  } else if (format == csv_format::RIS_SERVICE_VEHICLES) {
    utl::line_range{utl::buf_reader{file_content}}  //
        | utl::csv<ris_service_vehicles_row, Separator>()  //
        | utl::for_each([&](ris_service_vehicles_row const& row) {
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
  } else if (format == csv_format::FZG_KAP) {
    utl::line_range{utl::buf_reader{file_content}}  //
        | utl::csv<fzg_kap_row, Separator>()  //
        | utl::for_each([&](fzg_kap_row const& row) {
            auto& cap = caps.vehicle_capacity_map_[row.uic_number_.val()];
            cap.seats_ = row.seats_.val();
            ++res.loaded_entry_count_;
          });
  } else if (format == csv_format::FZG_GRUPPE) {
    utl::line_range{utl::buf_reader{file_content}}  //
        | utl::csv<fzg_gruppe_row, Separator>()  //
        |
        utl::for_each([&](fzg_gruppe_row const& row) {
          auto& cap = caps.vehicle_group_capacity_map_[row.group_name_->view()];
          cap.seats_ = row.seats_.val();
          cap.seats_1st_ = row.seats_1st_.val();
          cap.seats_2nd_ = row.seats_2nd_.val();
          cap.total_limit_ = row.critical_.val();
          ++res.loaded_entry_count_;
        });
  } else if (format == csv_format::GATTUNG) {
    utl::line_range{utl::buf_reader{file_content}}  //
        | utl::csv<gattung_row, Separator>()  //
        | utl::for_each([&](gattung_row const& row) {
            auto& cap = caps.gattung_capacity_map_[row.gattung_->view()];
            cap.seats_ = row.seats_.val();
            cap.seats_1st_ = row.seats_1st_.val();
            cap.seats_2nd_ = row.seats_2nd_.val();
            ++res.loaded_entry_count_;
          });
  } else if (format == csv_format::BAUREIHE) {
    utl::line_range{utl::buf_reader{file_content}}  //
        | utl::csv<baureihe_row, Separator>()  //
        | utl::for_each([&](baureihe_row const& row) {
            auto& cap = caps.baureihe_capacity_map_[row.vehicle_abr_->view()];
            cap.seats_ = row.seats_.val();
            cap.seats_1st_ = row.seats_1st_.val();
            cap.seats_2nd_ = row.seats_2nd_.val();
            cap.standing_ = row.standing_.val();
            cap.total_limit_ = row.total_limit_.val();
            ++res.loaded_entry_count_;
          });
  }

  return res;
}

load_capacities_result load_capacities(schedule const& sched,
                                       capacity_maps& caps,
                                       std::string_view const data) {
  auto file_content = utl::cstr{data};
  if (file_content.starts_with("\xEF\xBB\xBF")) {
    // skip utf-8 byte order mark (otherwise the first column is ignored)
    file_content = file_content.substr(3);
  }
  auto const format = get_csv_format(file_content.view());

  switch (format.separator_) {
    case csv_separator::COMMA:
      return load_capacities<','>(sched, caps, file_content, format.type_);
    case csv_separator::SEMICOLON:
      return load_capacities<';'>(sched, caps, file_content, format.type_);
  }
  throw utl::fail("paxmon: invalid csv separator in capacity file");
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
