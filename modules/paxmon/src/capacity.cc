#include "motis/paxmon/capacity.h"

#include <charconv>
#include <cstdint>
#include <ctime>
#include <algorithm>
#include <fstream>
#include <functional>
#include <iterator>
#include <optional>
#include <set>
#include <string>
#include <type_traits>
#include <utility>

#include "utl/parser/buf_reader.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/file.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/for_each.h"

#include "motis/core/common/logging.h"
#include "motis/core/access/station_access.h"
#include "motis/core/access/uuids.h"

#include "motis/paxmon/util/get_station_idx.h"

using namespace motis::logging;
using namespace motis::paxmon::util;

namespace motis::paxmon {

namespace {

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

enum class csv_format { TRIP, VEHICLE };

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

};  // namespace

std::size_t load_capacities(schedule const& sched,
                            std::string const& capacity_file,
                            capacity_maps& caps,
                            std::string const& match_log_file) {
  auto buf = utl::file(capacity_file.data(), "r").content();
  auto file_content = utl::cstr{buf.data(), buf.size()};
  if (file_content.starts_with("\xEF\xBB\xBF")) {
    // skip utf-8 byte order mark (otherwise the first column is ignored)
    file_content = file_content.substr(3);
  }
  auto const format = get_csv_format(file_content.view());
  auto entry_count = 0ULL;

  std::set<std::pair<std::string, std::string>> stations_not_found;

  if (format == csv_format::TRIP) {
    utl::line_range<utl::buf_reader>{file_content}  //
        | utl::csv<trip_row>()  //
        |
        utl::for_each([&](trip_row const& row) {
          if (row.train_nr_.val() != 0) {
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
              stations_not_found.insert(std::make_pair(
                  row.from_.val().to_str(), row.from_name_.val().to_str()));
            }
            if (row.to_.val() && to_station_idx == 0) {
              stations_not_found.insert(std::make_pair(
                  row.to_.val().to_str(), row.to_name_.val().to_str()));
            }
            if (departure == INVALID_TIME || arrival == INVALID_TIME) {
              return;
            }

            auto const tid = cap_trip_id{row.train_nr_.val(), from_station_idx,
                                         to_station_idx, departure, arrival};
            caps.trip_capacity_map_[tid] = row.seats_.val();
            ++entry_count;
          } else if (row.category_.val()) {
            caps.category_capacity_map_[row.category_.val().view()] =
                row.seats_.val();
            ++entry_count;
          }
        });
  } else if (format == csv_format::VEHICLE) {
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
            }
          });
  }

  if (!stations_not_found.empty()) {
    LOG(warn) << stations_not_found.size() << " stations not found";
    if (!match_log_file.empty()) {
      std::ofstream ml{match_log_file};
      ml << "stations not found:\n";
      for (auto const& [id, name] : stations_not_found) {
        ml << id << ": " << name << "\n";
      }
      LOG(warn) << "capacity match log report written to: " << match_log_file;
    }
  }

  return entry_count;
}

std::optional<std::pair<std::uint16_t, capacity_source>> get_trip_capacity(
    trip_capacity_map_t const& trip_map, trip const* trp,
    std::uint32_t train_nr) {
  auto const tid = cap_trip_id{train_nr, trp->id_.primary_.get_station_id(),
                               trp->id_.secondary_.target_station_id_,
                               trp->id_.primary_.get_time(),
                               trp->id_.secondary_.target_time_};
  if (auto const lb = trip_map.lower_bound(tid); lb != end(trip_map)) {
    if (lb->first == tid) {
      return {{lb->second, capacity_source::TRIP_EXACT}};
    } else if (lb->first.train_nr_ == train_nr) {
      return {{lb->second, capacity_source::TRAIN_NR}};
    } else if (lb != begin(trip_map)) {
      if (auto const prev = std::prev(lb);
          prev != end(trip_map) && prev->first.train_nr_ == train_nr) {
        return {{prev->second, capacity_source::TRAIN_NR}};
      }
    }
  }
  return {};
}

std::optional<std::uint32_t> get_line_nr(mcd::string const& line_id) {
  std::uint32_t line_nr = 0;
  auto const result =
      std::from_chars(line_id.data(), line_id.data() + line_id.size(), line_nr);
  if (result.ec == std::errc{} &&
      result.ptr == line_id.data() + line_id.size()) {
    return {line_nr};
  } else {
    return {};
  }
}

std::optional<std::pair<std::uint16_t, capacity_source>> get_trip_capacity(
    schedule const& sched, trip_capacity_map_t const& trip_map,
    category_capacity_map_t const& category_map, trip const* trp,
    connection_info const* ci, service_class const clasz) {

  auto const trp_train_nr = trp->id_.primary_.get_train_nr();
  if (auto const trip_capacity = get_trip_capacity(trip_map, trp, trp_train_nr);
      trip_capacity) {
    return trip_capacity;
  }

  if (ci->train_nr_ != trp_train_nr) {
    if (auto const trip_capacity =
            get_trip_capacity(trip_map, trp, ci->train_nr_);
        trip_capacity) {
      return trip_capacity;
    }
  }

  auto const trp_line_nr = get_line_nr(trp->id_.secondary_.line_id_);
  if (trp_line_nr && *trp_line_nr != trp_train_nr) {
    if (auto const trip_capacity =
            get_trip_capacity(trip_map, trp, *trp_line_nr);
        trip_capacity) {
      return trip_capacity;
    }
  }

  auto const ci_line_nr = get_line_nr(ci->line_identifier_);
  if (ci_line_nr && ci_line_nr != trp_line_nr && *ci_line_nr != trp_train_nr) {
    if (auto const trip_capacity =
            get_trip_capacity(trip_map, trp, *ci_line_nr);
        trip_capacity) {
      return trip_capacity;
    }
  }

  auto const& category = sched.categories_[ci->family_]->name_;
  if (auto const it = category_map.find(category); it != end(category_map)) {
    return {{it->second, capacity_source::CATEGORY}};
  } else if (auto const it = category_map.find(
                 std::to_string(static_cast<service_class_t>(clasz)));
             it != end(category_map)) {
    return {{it->second, capacity_source::CLASZ}};
  }

  return {};
}

std::optional<std::uint16_t> get_vehicle_capacity(capacity_maps const& caps,
                                                  std::uint64_t const uic) {
  if (auto const it = caps.vehicle_capacity_map_.find(uic);
      it != end(caps.vehicle_capacity_map_)) {
    return {it->second.limit()};
  } else {
    return {};
  }
}

std::uint16_t get_vehicles_capacity(capacity_maps const& caps,
                                    vehicle_order const& vo) {
  std::uint16_t cap = 0;
  // TODO(pablo): handle missing vehicle info
  for (auto const& uic : vo.uics_) {
    cap += get_vehicle_capacity(caps, uic).value_or(0);
  }
  return cap;
}

std::optional<std::pair<std::uint16_t, capacity_source>> get_section_capacity(
    schedule const& sched, capacity_maps const& caps, trip const* trp,
    ev_key const& ev_key_from, ev_key const& /*ev_key_to*/) {
  if (trp->uuid_.is_nil()) {
    return {};
  }
  if (auto const vo_it = caps.trip_vehicle_map_.find(trp->uuid_);
      vo_it != end(caps.trip_vehicle_map_)) {
    auto const& vo = vo_it->second;
    // try to match by event uuid
    auto const maybe_dep_uuid =
        motis::access::get_event_uuid(sched, trp, ev_key_from);
    if (maybe_dep_uuid.has_value()) {
      auto const dep_uuid = maybe_dep_uuid.value();
      if (auto const sec_vo =
              std::find_if(begin(vo.sections_), end(vo.sections_),
                           [&](auto const& sec_vo) {
                             return sec_vo.departure_uuid_ == dep_uuid;
                           });
          sec_vo != end(vo.sections_)) {
        return {{get_vehicles_capacity(caps, sec_vo->vehicles_),
                 capacity_source::TRIP_EXACT}};
      }
    }
    // departure eva fallback
    auto const dep_eva =
        sched.stations_.at(ev_key_from.get_station_idx())->eva_nr_;
    if (auto const sec_vo = std::find_if(begin(vo.sections_), end(vo.sections_),
                                         [&](auto const& sec_vo) {
                                           return sec_vo.departure_eva_ ==
                                                  dep_eva;
                                         });
        sec_vo != end(vo.sections_)) {
      return {{get_vehicles_capacity(caps, sec_vo->vehicles_),
               capacity_source::TRIP_EXACT}};
    }
    // TODO(pablo): station range fallback?
  }
  return {};
}

inline capacity_source get_worst_source(capacity_source const a,
                                        capacity_source const b) {
  return static_cast<capacity_source>(
      std::max(static_cast<std::underlying_type_t<capacity_source>>(a),
               static_cast<std::underlying_type_t<capacity_source>>(b)));
}

std::pair<std::uint16_t, capacity_source> get_capacity(
    schedule const& sched, light_connection const& lc,
    ev_key const& ev_key_from, ev_key const& ev_key_to,
    capacity_maps const& caps) {
  std::uint16_t capacity = 0;
  auto worst_source = capacity_source::TRIP_EXACT;

  auto ci = lc.full_con_->con_info_;
  for (auto const& trp : *sched.merged_trips_.at(lc.trips_)) {
    utl::verify(ci != nullptr, "get_capacity: missing connection_info");

    auto const section_capacity =
        get_section_capacity(sched, caps, trp, ev_key_from, ev_key_to);
    if (section_capacity.has_value()) {
      capacity += section_capacity->first;
      worst_source = get_worst_source(worst_source, section_capacity->second);
    } else {
      auto const trip_capacity = get_trip_capacity(
          sched, caps.trip_capacity_map_, caps.category_capacity_map_, trp, ci,
          lc.full_con_->clasz_);
      if (trip_capacity.has_value()) {
        capacity += trip_capacity->first;
        worst_source = get_worst_source(worst_source, trip_capacity->second);
      } else {
        return {UNKNOWN_CAPACITY, capacity_source::SPECIAL};
      }
    }

    ci = ci->merged_with_;
  }

  return {capacity, worst_source};
}

}  // namespace motis::paxmon
