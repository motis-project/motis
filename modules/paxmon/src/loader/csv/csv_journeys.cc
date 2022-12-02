#include "motis/paxmon/loader/csv/csv_journeys.h"

#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <optional>
#include <regex>
#include <sstream>
#include <string_view>
#include <utility>

#include "date/date.h"
#include "date/tz.h"

#include "fmt/ostream.h"

#include "utl/nwise.h"
#include "utl/parser/buf_reader.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/file.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/for_each.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "motis/core/common/logging.h"
#include "motis/core/schedule/time.h"
#include "motis/core/access/station_access.h"
#include "motis/core/access/trip_iterator.h"
#include "motis/core/conv/trip_conv.h"

#include "motis/paxmon/access/groups.h"
#include "motis/paxmon/compact_journey_util.h"
#include "motis/paxmon/loader/csv/motis_row.h"
#include "motis/paxmon/loader/csv/trek_row.h"
#include "motis/paxmon/tools/groups/group_generator.h"
#include "motis/paxmon/util/get_station_idx.h"
#include "motis/paxmon/util/interchange_time.h"

using namespace motis::logging;
using namespace motis::paxmon::util;
using namespace motis::paxmon::settings;
using namespace motis::paxmon::tools::groups;

template <>
struct fmt::formatter<std::optional<std::pair<std::uint64_t, std::uint64_t>>>
    : formatter<string_view> {
  template <typename FormatContext>
  auto format(std::optional<std::pair<std::uint64_t, std::uint64_t>> const& id,
              FormatContext& ctx) {
    if (id) {
      return formatter<string_view>::format(
          fmt::format("{}.{}", id->first, id->second), ctx);
    } else {
      return formatter<string_view>::format("-", ctx);
    }
  }
};

namespace motis::paxmon::loader::csv {

enum class csv_format { MOTIS, TREK };

struct trip_candidate {
  explicit operator bool() const { return trp_ != nullptr; }

  bool is_perfect_match() const {
    return trp_ != nullptr && enter_diff_ == 0 && exit_diff_ == 0 &&
           train_nr_match_;
  }

  bool is_better_than(trip_candidate const& other) const {
    if (!other) {
      return true;
    }
    if (train_nr_match_ && !other.train_nr_match_) {
      return true;
    } else if (!train_nr_match_ && other.train_nr_match_) {
      return false;
    }
    if (std::abs(travel_time_diff_) < std::abs(other.travel_time_diff_)) {
      return true;
    }
    return time_diff_cost() < other.time_diff_cost();
  }

  int time_diff_cost() const {
    return enter_diff_ * enter_diff_ + exit_diff_ * exit_diff_;
  }

  int travel_time() const { return static_cast<int>(exit_time_) - enter_time_; }

  trip* trp_{};
  time enter_time_{INVALID_TIME};
  time exit_time_{INVALID_TIME};
  std::uint32_t train_nr_{};
  bool train_nr_match_{};
  int enter_diff_{};
  int exit_diff_{};
  int travel_time_diff_{};
  std::string_view category_{};
};

std::pair<time, time> get_interval(time t, duration offset) {
  return {t > offset ? static_cast<time>(t - offset) : static_cast<time>(0),
          t > std::numeric_limits<time>::max() - offset
              ? std::numeric_limits<time>::max()
              : t + offset};
}

std::uint32_t get_train_nr(light_connection const* lc, std::uint32_t expected) {
  auto ci = lc->full_con_->con_info_;
  auto const first_train_nr = ci->train_nr_;
  for (; ci != nullptr; ci = ci->merged_with_) {
    if (ci->train_nr_ == expected) {
      return expected;
    }
  }
  return first_train_nr;
}

void enum_trip_candidates(schedule const& sched, std::uint32_t from_station_idx,
                          std::uint32_t to_station_idx, time enter_time,
                          time exit_time, std::uint32_t train_nr,
                          duration max_time_diff,
                          std::function<bool(trip_candidate&&)> const& cb) {
  auto const from_station = sched.station_nodes_.at(from_station_idx).get();
  auto const dep_interval = get_interval(enter_time, max_time_diff);
  auto const earliest_dep = dep_interval.first;
  auto const latest_dep = dep_interval.second;
  auto const expected_travel_time = static_cast<int>(exit_time - enter_time);
  auto keep_going = true;
  from_station->for_each_route_node([&](node const* route_node) {
    if (!keep_going) {
      return;
    }
    for (auto const& e : route_node->edges_) {
      if (e.type() != ::motis::edge::ROUTE_EDGE) {
        continue;
      }
      auto const& conns = e.m_.route_edge_.conns_;
      for (auto lc = std::lower_bound(begin(conns), end(conns),
                                      light_connection{earliest_dep});
           lc != end(conns) && lc->d_time_ <= latest_dep; lc = std::next(lc)) {
        if (lc->valid_ == 0) {
          continue;
        }
        auto const current_train_nr = get_train_nr(lc, train_nr);
        auto const& current_category =
            sched.categories_[lc->full_con_->con_info_->family_]->name_;
        auto const enter_diff = static_cast<int>(lc->d_time_) - enter_time;

        for (auto trp : *sched.merged_trips_.at(lc->trips_)) {
          for (auto const& stop : access::stops(trp)) {
            if (stop.get_station_id() == to_station_idx && stop.has_arrival()) {
              auto const arrival_time = stop.arr_lcon().a_time_;
              auto const exit_diff = static_cast<int>(arrival_time) - exit_time;
              auto const travel_time =
                  static_cast<int>(arrival_time) - lc->d_time_;
              if (travel_time < 0 || std::abs(exit_diff) > max_time_diff) {
                continue;
              }
              keep_going = cb(trip_candidate{
                  trp, lc->d_time_, arrival_time, current_train_nr,
                  current_train_nr == train_nr, enter_diff, exit_diff,
                  expected_travel_time - travel_time, current_category});
              if (!keep_going) {
                return;
              }
            }
          }
        }
      }
    }
  });
}

trip_candidate get_best_trip_candidate(schedule const& sched,
                                       std::uint32_t from_station_idx,
                                       std::uint32_t to_station_idx,
                                       time enter_time, time exit_time,
                                       std::uint32_t train_nr,
                                       duration max_time_diff) {
  auto best = trip_candidate{};

  enum_trip_candidates(sched, from_station_idx, to_station_idx, enter_time,
                       exit_time, train_nr, max_time_diff,
                       [&](trip_candidate&& candidate) {
                         if (candidate.is_better_than(best)) {
                           best = candidate;
                         }
                         return !best.is_perfect_match();
                       });

  return best;
}

void debug_trip_match(schedule const& sched, std::uint32_t from_station_idx,
                      std::uint32_t to_station_idx, time enter_time,
                      time exit_time, std::uint32_t train_nr,
                      std::string_view category, std::ofstream& match_log,
                      duration max_time_diff = 60) {
  auto const [earliest_dep, latest_dep] =
      get_interval(enter_time, max_time_diff);
  auto const expected_travel_time = static_cast<int>(exit_time - enter_time);
  match_log << "  stations: " << sched.stations_[from_station_idx]->name_
            << " => " << sched.stations_[to_station_idx]->name_ << "\n";
  match_log << "  possible matching trips with departure in ["
            << format_time(earliest_dep) << ", " << format_time(latest_dep)
            << "], expected travel_time=" << expected_travel_time << ":\n";

  enum_trip_candidates(
      sched, from_station_idx, to_station_idx, enter_time, exit_time, train_nr,
      max_time_diff, [&](trip_candidate&& candidate) {
        fmt::print(
            match_log,
            "    dep={} [{:+3}], train_nr={:6} [{}], "
            "category={:6} [{}] => arr={} [{:+3}], trip_train_nr={:6} [{}], "
            "tt={:3} [{:+3}]\n",
            format_time(candidate.enter_time_), candidate.enter_diff_,
            candidate.train_nr_, candidate.train_nr_match_ ? "✓" : "✗",
            candidate.category_, candidate.category_ == category ? "✓" : "✗",
            format_time(candidate.exit_time_), candidate.exit_diff_,
            candidate.trp_->id_.primary_.get_train_nr(),
            candidate.trp_->id_.primary_.get_train_nr() == train_nr ? "✓" : "✗",
            candidate.travel_time(), candidate.travel_time_diff_);
        return true;
      });

  match_log << "\n";
}

struct input_journey_leg {
  inline bool stations_found() const {
    return from_station_idx_ && to_station_idx_;
  }

  inline bool valid_times() const {
    return enter_time_ != INVALID_TIME && exit_time_ != INVALID_TIME;
  }

  inline bool trip_found() const { return trp_candidate_.trp_ != nullptr; }

  journey_leg to_journey_leg() const {
    utl::verify(stations_found(),
                "input_journey_leg.to_journey_leg(): stations not found");
    utl::verify(valid_times(),
                "input_journey_leg.to_journey_leg(): invalid times");
    utl::verify(trip_found(),
                "input_journey_leg.to_journey_leg(): trip not found");
    return journey_leg{trp_candidate_.trp_->trip_idx_,
                       from_station_idx_.value(),
                       to_station_idx_.value(),
                       enter_time_,
                       exit_time_,
                       enter_transfer_};
  }

  std::optional<std::uint32_t> from_station_idx_;
  std::optional<std::uint32_t> to_station_idx_;
  time enter_time_{INVALID_TIME};
  time exit_time_{INVALID_TIME};
  trip_candidate trp_candidate_;
  std::optional<transfer_info> enter_transfer_;
};

void set_transfer_info(schedule const& sched,
                       std::vector<input_journey_leg>& partial_journey) {
  if (partial_journey.size() < 2) {
    return;
  }
  auto const& prev_leg = partial_journey[partial_journey.size() - 2];
  if (!prev_leg.to_station_idx_ || prev_leg.exit_time_ == INVALID_TIME) {
    return;
  }
  auto& cur_leg = partial_journey.back();
  auto const arrival_station = prev_leg.to_station_idx_.value();
  auto const arrival_section =
      get_arrival_section(sched, prev_leg.trp_candidate_.trp_, arrival_station,
                          prev_leg.exit_time_);
  auto const departure_station = cur_leg.from_station_idx_.value();
  auto const departure_section =
      get_departure_section(sched, cur_leg.trp_candidate_.trp_,
                            departure_station, cur_leg.enter_time_);
  if (arrival_section && departure_section) {
    cur_leg.enter_transfer_ =
        util::get_transfer_info(sched, *arrival_section, *departure_section);
  } else {
    if (prev_leg.trp_candidate_.trp_ != nullptr &&
        cur_leg.trp_candidate_.trp_ != nullptr) {
      LOG(warn) << "trip sections not found during transfer check: arrival="
                << (arrival_section ? "found" : "not found")
                << ", departure=" << (departure_section ? "found" : "not found")
                << ", arrival trip=" << prev_leg.trp_candidate_.trp_
                << " (perfect match="
                << prev_leg.trp_candidate_.is_perfect_match()
                << "), departure trip=" << cur_leg.trp_candidate_.trp_
                << " (perfect match="
                << cur_leg.trp_candidate_.is_perfect_match() << ")";
    }
    cur_leg.enter_transfer_ = util::get_transfer_info(
        sched, arrival_station, {}, departure_station, {});
  }
}

void write_match_log(
    std::ofstream& match_log, schedule const& sched,
    input_journey_leg const& leg,
    std::optional<std::pair<std::uint64_t, std::uint64_t>> const& current_id,
    motis_row const& row,
    std::vector<input_journey_leg> const& current_input_legs,
    duration const debug_match_tolerance) {
  if (!match_log) {
    return;
  }
  if (!leg.stations_found()) {
    if (!leg.from_station_idx_) {
      fmt::print(match_log, "[{}] Station not found: {}\n", current_id,
                 row.from_.val().view());
    }
    if (!leg.to_station_idx_) {
      fmt::print(match_log, "[{}] Station not found: {}\n", current_id,
                 row.to_.val().view());
    }
  }
  if (!leg.valid_times()) {
    if (leg.enter_time_ == INVALID_TIME) {
      fmt::print(match_log, "[{}] Invalid enter timestamp: {}\n", current_id,
                 format_unix_time(row.enter_.val()));
    }
    if (leg.exit_time_ == INVALID_TIME) {
      fmt::print(match_log, "[{}] Invalid exit timestamp: {}\n", current_id,
                 format_unix_time(row.exit_.val()));
    }
  }
  if (!leg.trip_found()) {
    fmt::print(match_log,
               "[{}] Trip not found: from={:7}, to={:7}, enter={}, "
               "exit={}, train_nr={:6}, category={:6}, leg={}\n",
               current_id, row.from_.val().view(), row.to_.val().view(),
               format_unix_time(row.enter_.val()),
               format_unix_time(row.exit_.val()), row.train_nr_.val(),
               row.category_.val().view(), current_input_legs.size());
    if (leg.stations_found() && leg.valid_times()) {
      debug_trip_match(
          sched, leg.from_station_idx_.value(), leg.to_station_idx_.value(),
          leg.enter_time_, leg.exit_time_, row.train_nr_.val(),
          row.category_.val().view(), match_log, debug_match_tolerance);
    }
  }
}

void write_match_log(
    std::ofstream& match_log, schedule const& sched,
    input_journey_leg const& leg,
    std::optional<std::pair<std::uint64_t, std::uint64_t>> const& current_id,
    trek_row const& row,
    std::vector<input_journey_leg> const& current_input_legs,
    duration const debug_match_tolerance) {
  if (!match_log) {
    return;
  }
  if (!leg.stations_found()) {
    if (!leg.from_station_idx_) {
      fmt::print(match_log, "[{}] Station not found: {}\n", current_id,
                 row.from_.val().view());
    }
    if (!leg.to_station_idx_) {
      fmt::print(match_log, "[{}] Station not found: {}\n", current_id,
                 row.to_.val().view());
    }
  }
  if (!leg.valid_times()) {
    if (leg.enter_time_ == INVALID_TIME) {
      fmt::print(match_log, "[{}] Invalid enter timestamp: {}\n", current_id,
                 row.enter_.val().view());
    }
    if (leg.exit_time_ == INVALID_TIME) {
      fmt::print(match_log, "[{}] Invalid exit timestamp: {}\n", current_id,
                 row.exit_.val().view());
    }
  }
  if (!leg.trip_found()) {
    fmt::print(match_log,
               "[{}] Trip not found: from={:7}, to={:7}, enter={}, "
               "exit={}, train_nr={:6}, category={:6}, leg={}\n",
               current_id, row.from_.val().view(), row.to_.val().view(),
               row.enter_.val().view(), row.exit_.val().view(),
               row.train_nr_.val(), row.category_.val().view(),
               current_input_legs.size());
    if (leg.stations_found() && leg.valid_times()) {
      debug_trip_match(
          sched, leg.from_station_idx_.value(), leg.to_station_idx_.value(),
          leg.enter_time_, leg.exit_time_, row.train_nr_.val(),
          row.category_.val().view(), match_log, debug_match_tolerance);
    }
  }
}

csv_format get_csv_format(std::string_view const file_content) {
  if (auto const nl = file_content.find('\n'); nl != std::string_view::npos) {
    auto const header = file_content.substr(0, nl);
    if (header.find("leg_type") != std::string_view::npos) {
      utl::verify(
          header.find(',') != std::string_view::npos,
          "paxmon: only ',' separator supported for motis csv journey files");
      return csv_format::MOTIS;
    } else if (header.find("EinZeitpunkt") != std::string_view::npos) {
      utl::verify(header.find(';') != std::string_view::npos,
                  "paxmon: only ';' separator supported for daily trek files");
      return csv_format::TREK;
    } else {
      throw utl::fail("paxmon: unsupported csv journey input format");
    }
  }
  throw utl::fail("paxmon: empty journey input file");
}

time parse_trek_timestamp(std::string_view const val, date::time_zone const* tz,
                          schedule const& sched) {
  auto ss = std::stringstream{};
  auto ls = date::local_seconds{};
  ss << val;
  ss >> date::parse("%d.%m.%Y %H:%M:%S", ls);
  if (ss.fail()) {
    return INVALID_TIME;
  }
  auto const zoned = date::make_zoned(tz, ls);
  auto const ts = zoned.get_sys_time();
  auto unix_ts =
      std::chrono::duration_cast<std::chrono::seconds>(ts.time_since_epoch())
          .count();
  return unix_to_motistime(sched.schedule_begin_, unix_ts);
}

loader_result load_journeys(schedule const& sched, universe& uv,
                            capacity_maps const& caps,
                            std::string const& journey_file,
                            journey_input_settings const& settings) {
  auto const match_tolerance = settings.match_tolerance_;
  auto const debug_match_tolerance = match_tolerance + 60;
  auto const split_groups = settings.split_groups_;
  auto result = loader_result{};
  auto journeys_with_invalid_legs = 0ULL;
  auto journeys_with_no_valid_legs = 0ULL;
  auto journeys_with_inexact_matches = 0ULL;
  auto journeys_with_missing_trips = 0ULL;
  auto journeys_with_invalid_transfer_times = 0ULL;
  auto journeys_too_long = 0ULL;

  auto buf = utl::file(journey_file.data(), "r").content();
  auto file_content = utl::cstr{buf.data(), buf.size()};
  if (file_content.starts_with("\xEF\xBB\xBF")) {
    // skip utf-8 byte order mark (otherwise the first column is ignored)
    file_content = file_content.substr(3);
  }

  auto const format = get_csv_format(file_content.view());

  std::ofstream match_log;
  if (!settings.journey_match_log_file_.empty()) {
    match_log.open(settings.journey_match_log_file_);
  }

  auto group_gen = group_generator{settings.split_groups_size_mean_,
                                   settings.split_groups_size_stddev_, 0, 1,
                                   settings.split_groups_seed_};

  using id_t = std::pair<std::uint64_t, std::uint64_t>;
  auto current_id = std::optional<id_t>{};
  auto current_input_legs = std::vector<input_journey_leg>{};
  std::uint16_t current_passengers = 0;

  auto const add_journey = [&](std::size_t start_idx, std::size_t end_idx,
                               route_source_flags source_flags) {
    if (start_idx == end_idx) {
      return;
    }
    auto source =
        data_source{current_id.value().first, current_id.value().second};
    auto const inexact_time = std::any_of(
        std::next(begin(current_input_legs), start_idx),
        std::next(begin(current_input_legs), end_idx),
        [](auto const& leg) { return !leg.trp_candidate_.is_perfect_match(); });
    if (inexact_time) {
      source_flags |= route_source_flags::MATCH_INEXACT_TIME;
      ++journeys_with_inexact_matches;
    }
    auto const all_trips_found =
        std::all_of(std::next(begin(current_input_legs), start_idx),
                    std::next(begin(current_input_legs), end_idx),
                    [](auto const& leg) { return leg.trip_found(); });

    // TODO(pablo): is this the best way to handle invalid transfer times?
    auto invalid_transfer_times = false;
    for (auto const& [l1, l2] :
         utl::nwise_range<2, decltype(begin(current_input_legs))>{
             std::next(begin(current_input_legs), start_idx),
             std::next(begin(current_input_legs), end_idx)}) {
      if (l2.enter_time_ < l1.exit_time_ ||
          (l2.enter_transfer_.has_value() && ((l2.enter_time_ - l1.exit_time_) <
                                              l2.enter_transfer_->duration_))) {
        invalid_transfer_times = true;
      }
    }

    if (all_trips_found && !invalid_transfer_times) {
      ++result.loaded_journeys_;
      auto current_journey = compact_journey{};
      current_journey.legs_ =
          utl::to_vec(std::next(begin(current_input_legs), start_idx),
                      std::next(begin(current_input_legs), end_idx),
                      [&](auto const& leg) { return leg.to_journey_leg(); });
      utl::verify(!current_journey.legs_.empty(), "empty csv journey");
      current_journey.legs_.front().enter_transfer_ = {};
      if (current_journey.scheduled_duration() > 24 * 60) {
        ++journeys_too_long;
        return;
      }
      auto tpg = temp_passenger_group{
          0,
          source,
          current_passengers,
          {{0, 1.0F, current_journey, current_journey.scheduled_arrival_time(),
            0, source_flags, true}}};
      if (split_groups) {
        tpg.source_.secondary_ref_ *= 100;
        auto distributed = 0U;
        while (distributed < current_passengers) {
          auto const group_size =
              group_gen.get_group_size(current_passengers - distributed);
          ++tpg.source_.secondary_ref_;
          distributed += group_size;
          tpg.passengers_ = group_size;
          add_passenger_group(uv, sched, caps, tpg, false);
        }
      } else {
        add_passenger_group(uv, sched, caps, tpg, false);
      }
    } else {
      if (!all_trips_found) {
        ++journeys_with_missing_trips;
      }
      if (invalid_transfer_times) {
        ++journeys_with_invalid_transfer_times;
      }
      auto const& first_leg = current_input_legs.at(start_idx);
      auto const& last_leg = current_input_legs.at(end_idx - 1);
      if (split_groups) {
        source.secondary_ref_ *= 100;
        auto distributed = 0U;
        while (distributed < current_passengers) {
          auto const group_size =
              group_gen.get_group_size(current_passengers - distributed);
          ++source.secondary_ref_;
          distributed += group_size;
          result.unmatched_journeys_.emplace_back(
              unmatched_journey{first_leg.from_station_idx_.value(),
                                last_leg.to_station_idx_.value(),
                                first_leg.enter_time_, source, group_size});
        }
      } else {
        result.unmatched_journeys_.emplace_back(unmatched_journey{
            first_leg.from_station_idx_.value(),
            last_leg.to_station_idx_.value(), first_leg.enter_time_, source,
            current_passengers});
      }
    }
  };

  auto const finish_journey = [&]() {
    if (!current_id || current_input_legs.empty()) {
      return;
    }
    auto source_flags = route_source_flags::NONE;
    auto const possible_leg_count =
        std::count_if(begin(current_input_legs), end(current_input_legs),
                      [](auto const& leg) {
                        return leg.stations_found() && leg.valid_times();
                      });
    if (possible_leg_count == 0) {
      ++journeys_with_no_valid_legs;
      return;
    } else if (possible_leg_count < current_input_legs.size()) {
      ++journeys_with_invalid_legs;
      source_flags |= route_source_flags::MATCH_JOURNEY_SUBSET;
    }

    auto subset_start = 0ULL;
    for (auto i = 0ULL; i < current_input_legs.size(); ++i) {
      auto const& leg = current_input_legs[i];
      if (!leg.stations_found() || !leg.valid_times()) {
        add_journey(subset_start, i, source_flags);
        subset_start = i + 1;
      }
    }
    add_journey(subset_start, current_input_legs.size(), source_flags);
  };

  if (format == csv_format::MOTIS) {
    utl::line_range<utl::buf_reader>{file_content}  //
        | utl::csv<motis_row>()  //
        | utl::for_each([&](motis_row const& row) {
            auto const id = id_t{row.id_.val(), row.secondary_id_.val()};
            if (id != current_id) {
              finish_journey();
              current_id = id;
              current_passengers = row.passengers_.val();
              current_input_legs.clear();
            }
            if (row.leg_type_.val() == "FOOT") {
              return;
            }
            auto& leg = current_input_legs.emplace_back();
            leg.from_station_idx_ =
                get_station_idx(sched, row.from_.val().view());
            leg.to_station_idx_ = get_station_idx(sched, row.to_.val().view());
            leg.enter_time_ =
                unix_to_motistime(sched.schedule_begin_, row.enter_.val());
            leg.exit_time_ =
                unix_to_motistime(sched.schedule_begin_, row.exit_.val());

            if (leg.stations_found() && leg.valid_times()) {
              leg.trp_candidate_ = get_best_trip_candidate(
                  sched, leg.from_station_idx_.value(),
                  leg.to_station_idx_.value(), leg.enter_time_, leg.exit_time_,
                  row.train_nr_.val(), match_tolerance);
              set_transfer_info(sched, current_input_legs);
            }
            write_match_log(match_log, sched, leg, current_id, row,
                            current_input_legs, debug_match_tolerance);
          });
  } else if (format == csv_format::TREK) {
    auto const tz = settings.journey_timezone_.empty()
                        ? date::current_zone()
                        : date::locate_zone(settings.journey_timezone_);
    utl::line_range<utl::buf_reader>{file_content}  //
        | utl::csv<trek_row, ';'>()  //
        | utl::for_each([&](trek_row const& row) {
            auto const base_id = id_t{row.id_.val(), 0U};
            if (base_id != current_id) {
              finish_journey();
              current_id = base_id;
              current_passengers = row.passengers_.val();
              current_input_legs.clear();
            }
            if (row.category_.val() == "Fussweg") {
              return;
            }
            auto& leg = current_input_legs.emplace_back();
            leg.from_station_idx_ =
                get_station_idx(sched, row.from_.val().view());
            leg.to_station_idx_ = get_station_idx(sched, row.to_.val().view());
            leg.enter_time_ =
                parse_trek_timestamp(row.enter_.val().view(), tz, sched);
            leg.exit_time_ =
                parse_trek_timestamp(row.exit_.val().view(), tz, sched);

            if (leg.stations_found() && leg.valid_times()) {
              leg.trp_candidate_ = get_best_trip_candidate(
                  sched, leg.from_station_idx_.value(),
                  leg.to_station_idx_.value(), leg.enter_time_, leg.exit_time_,
                  row.train_nr_.val(), match_tolerance);
              set_transfer_info(sched, current_input_legs);
            }
            write_match_log(match_log, sched, leg, current_id, row,
                            current_input_legs, debug_match_tolerance);
          });
  }

  finish_journey();

  LOG(info) << "loaded " << result.loaded_journeys_ << " journeys";
  LOG(info) << journeys_with_invalid_legs << " journeys with some invalid legs";
  LOG(info) << journeys_with_no_valid_legs << " journeys with no valid legs";
  LOG(info) << journeys_with_inexact_matches
            << " journeys with inexact matches";
  LOG(info) << journeys_with_missing_trips << " journeys with missing trips";
  LOG(info) << journeys_with_invalid_transfer_times
            << " journeys with invalid transfer times";
  LOG(info) << journeys_too_long << " journeys that are too long (skipped)";
  LOG(info) << result.unmatched_journeys_.size() << " unmatched journeys";

  return result;
}

}  // namespace motis::paxmon::loader::csv
