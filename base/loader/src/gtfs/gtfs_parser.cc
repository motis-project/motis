// clang-tidy crashes while processing this file
// NOLINTBEGIN(bugprone-unchecked-optional-access)

#include "motis/loader/gtfs/gtfs_parser.h"

#include <chrono>
#include <filesystem>
#include <numeric>

#include "boost/algorithm/string.hpp"
#include "boost/date_time/posix_time/posix_time.hpp"

#include "utl/erase_if.h"
#include "utl/get_or_create.h"
#include "utl/pairwise.h"
#include "utl/parallel_for.h"
#include "utl/parser/cstr.h"
#include "utl/pipes/accumulate.h"
#include "utl/pipes/all.h"
#include "utl/pipes/remove_if.h"
#include "utl/pipes/transform.h"
#include "utl/pipes/vec.h"
#include "utl/progress_tracker.h"

#include "cista/hash.h"
#include "cista/mmap.h"

#include "geo/latlng.h"
#include "geo/point_rtree.h"

#include "motis/core/common/constants.h"
#include "motis/core/common/date_time_util.h"
#include "motis/core/common/logging.h"
#include "motis/core/schedule/time.h"
#include "motis/loader/gtfs/agency.h"
#include "motis/loader/gtfs/calendar.h"
#include "motis/loader/gtfs/calendar_date.h"
#include "motis/loader/gtfs/feed_info.h"
#include "motis/loader/gtfs/files.h"
#include "motis/loader/gtfs/route.h"
#include "motis/loader/gtfs/services.h"
#include "motis/loader/gtfs/stop.h"
#include "motis/loader/gtfs/stop_time.h"
#include "motis/loader/gtfs/transfers.h"
#include "motis/loader/gtfs/trip.h"
#include "motis/loader/util.h"
#include "motis/schedule-format/Schedule_generated.h"

namespace fbs64 = flatbuffers64;
namespace fs = std::filesystem;
using namespace motis::logging;
using std::get;

namespace motis::loader::gtfs {

auto const required_files = {AGENCY_FILE, STOPS_FILE, ROUTES_FILE, TRIPS_FILE,
                             STOP_TIMES_FILE};

cista::hash_t hash(fs::path const& path) {
  auto hash = cista::BASE_HASH;
  auto const hash_file = [&](fs::path const& p) {
    if (!fs::is_regular_file(p)) {
      return;
    }
    cista::mmap m{p.generic_string().c_str(), cista::mmap::protection::READ};
    hash = cista::hash_combine(
        cista::hash(std::string_view{
            reinterpret_cast<char const*>(m.begin()),
            std::min(static_cast<size_t>(50 * 1024 * 1024), m.size())}),
        hash);
  };

  for (auto const& file_name : required_files) {
    hash_file(path / file_name);
  }
  hash_file(path / CALENDAR_FILE);
  hash_file(path / CALENDAR_DATES_FILE);

  return hash;
}

bool gtfs_parser::applicable(fs::path const& path) {
  for (auto const& file_name : required_files) {
    if (!fs::is_regular_file(path / file_name)) {
      return false;
    }
  }

  return fs::is_regular_file(path / CALENDAR_FILE) ||
         fs::is_regular_file(path / CALENDAR_DATES_FILE);
}

std::vector<std::string> gtfs_parser::missing_files(
    fs::path const& path) const {
  std::vector<std::string> files;
  if (!fs::is_directory(path)) {
    files.emplace_back(path.string());
  }

  std::copy_if(
      begin(required_files), end(required_files), std::back_inserter(files),
      [&](std::string const& f) { return !fs::is_regular_file(path / f); });

  if (!fs::is_regular_file(path / CALENDAR_FILE) &&
      !fs::is_regular_file(path / CALENDAR_DATES_FILE)) {
    files.emplace_back(CALENDAR_FILE);
    files.emplace_back(CALENDAR_DATES_FILE);
  }

  return files;
}

std::time_t to_unix_time(date::sys_days const& date) {
  return std::chrono::time_point_cast<std::chrono::seconds>(date)
      .time_since_epoch()
      .count();
}

void fix_flixtrain_transfers(trip_map& trips,
                             std::map<stop_pair, transfer>& transfers) {
  for (auto const& id_prefix : {"FLIXBUS:FLX", "FLIXBUS:K"}) {
    for (auto it = trips.lower_bound(id_prefix);
         it != end(trips) && boost::starts_with(it->first, id_prefix); ++it) {
      for (auto const& [dep_entry, arr_entry] :
           utl::pairwise(it->second->stop_times_)) {
        auto& dep = dep_entry.second;
        auto& arr = arr_entry.second;

        if (dep.stop_ == nullptr) {
          continue;  // already gone
        }

        auto const& dep_name = dep.stop_->name_;
        auto const& arr_name = arr.stop_->name_;
        if (utl::get_until(utl::cstr{dep_name}, ',') !=
            utl::get_until(utl::cstr{arr_name}, ',')) {
          continue;  // different towns
        }

        // normal case: bus stop after train stop
        auto const arr_duplicate =
            static_cast<bool>(boost::ifind_first(dep_name, "train")) &&
            !static_cast<bool>(boost::ifind_first(arr_name, "train")) &&
            dep.dep_.time_ == arr.arr_.time_ &&
            arr.arr_.time_ == arr.dep_.time_;

        // may happen on last stop: train stop after bus_stop
        auto const dep_duplicate =
            !static_cast<bool>(boost::ifind_first(dep_name, "train")) &&
            static_cast<bool>(boost::ifind_first(arr_name, "train")) &&
            dep.dep_.time_ == arr.arr_.time_ &&
            dep.arr_.time_ == dep.dep_.time_;

        if (arr_duplicate || dep_duplicate) {
          auto dur = static_cast<int>(
              geo::distance(dep.stop_->coord_, arr.stop_->coord_) / WALK_SPEED /
              60);
          transfers.insert(
              {{dep.stop_, arr.stop_}, {dur, transfer::GENERATED}});
          transfers.insert(
              {{arr.stop_, dep.stop_}, {dur, transfer::GENERATED}});
        }

        if (arr_duplicate) {
          arr.stop_ = nullptr;
        }
        if (dep_duplicate) {
          dep.stop_ = nullptr;
        }
      }

      utl::erase_if(it->second->stop_times_,
                    [](auto const& s) { return s.second.stop_ == nullptr; });
    }
  }
}

void gtfs_parser::parse(parser_options const& opt, fs::path const& root,
                        fbs64::FlatBufferBuilder& fbb) {
  motis::logging::scoped_timer const global_timer{"gtfs parser"};

  auto const load = [&](char const* file) {
    return fs::is_regular_file(root / file) ? loaded_file{root / file}
                                            : loaded_file{};
  };
  auto const feeds = read_feed_publisher(load(FEED_INFO_FILE));
  auto const agencies = read_agencies(load(AGENCY_FILE));
  auto const stops = read_stops(load(STOPS_FILE));
  auto const routes = read_routes(load(ROUTES_FILE), agencies);
  auto const calendar = read_calendar(load(CALENDAR_FILE));
  auto const dates = read_calendar_date(load(CALENDAR_DATES_FILE));
  auto const service = merge_traffic_days(calendar, dates);
  auto transfers = read_transfers(load(TRANSFERS_FILE), stops);
  auto [trips, blocks] = read_trips(load(TRIPS_FILE), routes, service);
  read_frequencies(load(FREQUENCIES_FILE), trips);
  read_stop_times(load(STOP_TIMES_FILE), trips, stops);
  fix_flixtrain_transfers(trips, transfers);
  for (auto& [_, trip] : trips) {
    trip->interpolate();
  }

  LOG(logging::info) << "read " << trips.size() << " trips, " << routes.size()
                     << " routes";

  std::map<category, fbs64::Offset<Category>> fbs_categories;
  std::map<agency const*, fbs64::Offset<Provider>> fbs_providers;
  std::map<std::string, fbs64::Offset<fbs64::String>> fbs_strings;
  std::map<stop const*, fbs64::Offset<Station>> fbs_stations;
  std::map<trip::stop_seq, fbs64::Offset<Route>> fbs_routes;
  std::map<trip::stop_seq_numbers, fbs64::Offset<fbs64::Vector<uint32_t>>>
      fbs_seq_numbers;
  std::vector<fbs64::Offset<Service>> const fbs_services;

  auto get_or_create_stop = [&](stop const* s) {
    return utl::get_or_create(fbs_stations, s, [&]() {
      return CreateStation(
          fbb, fbb.CreateString(s->id_), fbb.CreateString(s->name_),
          s->coord_.lat_, s->coord_.lng_, 2,
          fbb.CreateVector(std::vector<fbs64::Offset<fbs64::String>>()), 0,
          s->timezone_.empty() ? 0 : fbb.CreateString(s->timezone_));
    });
  };

  auto get_or_create_category = [&](trip const* t) {
    auto const* r = t->route_;
    if (auto cat = r->get_category(); cat.has_value()) {
      auto const create = [&]() {
        return utl::get_or_create(fbs_categories, *cat, [&]() {
          return CreateCategory(fbb, fbb.CreateString(cat->name_),
                                cat->output_rule_ & category::output::BASE);
        });
      };

      if (cat->name_ == "DPN") {
        if (t->avg_speed() > 100) {
          cat->name_ = "High Speed Rail";
          return create();
        } else if (t->distance() > 400) {
          cat->name_ = "Long Distance Trains";
          return create();
        }
      } else if (cat->name_ == "Bus" && t->distance() > 100) {
        cat->output_rule_ =
            category::output::FORCE_PROVIDER_INSTEAD_OF_CATEGORY;
        cat->name_ = "Coach";
        return create();
      }

      if ((cat->output_rule_ &
           category::output::ROUTE_NAME_SHORT_INSTEAD_OF_CATEGORY) ==
          category::output::ROUTE_NAME_SHORT_INSTEAD_OF_CATEGORY) {
        auto const is_number =
            std::all_of(begin(r->short_name_), end(r->short_name_),
                        [](auto c) { return std::isdigit(c); });
        cat->name_ = is_number ? cat->name_ : r->short_name_;
      }
      return create();
    } else {
      auto desc = r->desc_;
      static auto const short_tags =
          std::map<std::string, std::string>{{"RegioExpress", "RE"},
                                             {"Regionalzug", "RZ"},
                                             {"InterRegio", "IR"},
                                             {"Intercity", "IC"}};
      if (auto it = short_tags.find(desc); it != end(short_tags)) {
        desc = it->second;
      }
      return utl::get_or_create(fbs_categories, category{desc, 0}, [&]() {
        return CreateCategory(fbb, fbb.CreateString(desc), 0);
      });
    }
  };

  auto get_or_create_provider = [&](agency const* a) {
    return utl::get_or_create(fbs_providers, a, [&]() {
      if (a == nullptr) {
        auto const name = fbb.CreateString("UNKNOWN_AGENCY");
        return CreateProvider(fbb, name, name, name, 0);
      }

      auto const name = fbb.CreateString(a->name_);
      return CreateProvider(
          fbb, fbb.CreateString(a->id_), name, name,
          a->timezone_.empty() ? 0 : fbb.CreateString(a->timezone_));
    });
  };

  auto get_or_create_str = [&](std::string const& s) {
    return utl::get_or_create(fbs_strings, s,
                              [&]() { return fbb.CreateString(s); });
  };

  auto get_or_create_direction = [&](trip const* t) {
    if (!t->headsign_.empty()) {
      return get_or_create_str(t->headsign_);
    } else {
      return get_or_create_str(t->stops().back().stop_->name_);
    }
  };

  motis::logging::scoped_timer const export_timer{"export"};
  auto progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Export schedule.raw")
      .out_bounds(60.F, 100.F)
      .in_high(trips.size());
  auto const interval = Interval{to_unix_time(service.first_day_),
                                 to_unix_time(service.last_day_)};

  auto n_services = 0U;
  auto const trips_file =
      fbb.CreateString((root / STOP_TIMES_FILE).generic_string());
  auto const create_service =
      [&](trip* t, bitfield const& traffic_days,
          bool const is_rule_service_participant,
          ScheduleRelationship const schedule_relationship) {
        auto const is_train_number = [](auto const& s) {
          return !s.empty() &&
                 std::all_of(begin(s), end(s),
                             [](auto&& c) -> bool { return std::isdigit(c); });
        };

        auto const day_offset = t->stop_times_.front().dep_.time_ / 1440;
        auto const adjusted_traffic_days = traffic_days << day_offset;

        int train_nr = 0;
        if (is_train_number(t->short_name_)) {
          train_nr = std::stoi(t->short_name_);
        } else if (is_train_number(t->headsign_)) {
          train_nr = std::stoi(t->headsign_);
        }

        ++n_services;
        auto const stop_seq = t->stops();
        return CreateService(
            fbb,
            utl::get_or_create(
                fbs_routes, stop_seq,
                [&]() {
                  return CreateRoute(
                      fbb,  //
                      fbb.CreateVector(
                          utl::to_vec(stop_seq,
                                      [&](trip::stop_identity const& s) {
                                        return get_or_create_stop(s.stop_);
                                      })),
                      fbb.CreateVector(utl::to_vec(
                          stop_seq,
                          [](trip::stop_identity const& s) {
                            return static_cast<uint8_t>(s.in_allowed_ ? 1U
                                                                      : 0U);
                          })),
                      fbb.CreateVector(utl::to_vec(
                          stop_seq, [](trip::stop_identity const& s) {
                            return static_cast<uint8_t>(s.out_allowed_ ? 1U
                                                                       : 0U);
                          })));
                }),
            fbb.CreateString(serialize_bitset(adjusted_traffic_days)),
            fbb.CreateVector(repeat_n(
                CreateSection(
                    fbb, get_or_create_category(t),
                    get_or_create_provider(t->route_->agency_), train_nr,
                    get_or_create_str(t->route_->short_name_),
                    fbb.CreateVector(std::vector<fbs64::Offset<Attribute>>()),
                    CreateDirection(fbb, 0, get_or_create_direction(t))),
                stop_seq.size() - 1)),
            0 /* tracks currently not extracted */,
            fbb.CreateVector(utl::all(t->stop_times_)  //
                             | utl::accumulate(
                                   [&](std::vector<int>&& times,
                                       flat_map<stop_time>::entry_t const& st) {
                                     times.emplace_back(st.second.arr_.time_ -
                                                        day_offset * 1440);
                                     times.emplace_back(st.second.dep_.time_ -
                                                        day_offset * 1440);
                                     return std::move(times);
                                   },
                                   std::vector<int>())),
            0 /* route key obsolete */,
            CreateServiceDebugInfo(fbb, trips_file, t->from_line_, t->to_line_),
            is_rule_service_participant, 0 /* initial train number */,
            get_or_create_str(t->id_),
            utl::get_or_create(
                fbs_seq_numbers, t->seq_numbers(),
                [&]() { return fbb.CreateVector(t->seq_numbers()); }),
            schedule_relationship);
      };

  auto output_services =
      utl::all(trips)  //
      | utl::remove_if([&](auto const& entry) {
          progress_tracker->increment();
          auto const stop_count = entry.second->stops().size();
          if (stop_count < 2) {
            LOG(warn) << "invalid trip " << entry.first << ": "
                      << entry.second->stops().size() << " stops";
          }
          return stop_count < 2;
        })  //
      | utl::remove_if([&](auto const& entry) {
          // Frequency services are written separately.
          return entry.second->frequency_.has_value();
        })  //
      | utl::transform([&](auto const& entry) {
          auto const t = entry.second.get();
          return create_service(
              entry.second.get(), *t->service_,
              t->block_ != nullptr && t->block_->trips_.size() >= 2,
              ScheduleRelationship_SCHEDULED);  //
        })  //
      | utl::vec();

  for (auto const& [id, t] : trips) {
    if (t->frequency_.has_value()) {
      t->expand_frequencies(
          [&](trip& x, ScheduleRelationship const schedule_relationship) {
            output_services.emplace_back(
                create_service(&x, *x.service_, false, schedule_relationship));
          });
    }
  }

  std::vector<fbs64::Offset<RuleService>> rule_services;
  for (auto const& blk : blocks) {
    for (auto const& [trips, traffic_days] : blk.second->rule_services()) {
      auto const td = traffic_days;

      if (trips.size() == 1) {
        output_services.emplace_back(
            create_service(trips.front(), traffic_days, false,
                           ScheduleRelationship_SCHEDULED));
        continue;
      }

      std::vector<fbs64::Offset<Rule>> rules;
      std::map<trip*, fbs64::Offset<Service>> services;
      for (auto const& p : utl::pairwise(trips)) {
        auto const& a = get<0>(p);
        auto const& b = get<1>(p);
        auto const transition_stop =
            get_or_create_stop(a->stop_times_.back().stop_);
        auto const base_offset_a = a->stop_times_.front().dep_.time_ / 1440;
        rules.emplace_back(CreateRule(
            fbb, RuleType_THROUGH,
            utl::get_or_create(services, a,
                               [&] {
                                 return create_service(
                                     a, td, true,
                                     ScheduleRelationship_SCHEDULED);
                               }),
            utl::get_or_create(services, b,
                               [&] {
                                 return create_service(
                                     b, td, true,
                                     ScheduleRelationship_SCHEDULED);
                               }),
            transition_stop, transition_stop,
            (a->stop_times_.back().arr_.time_ / 1440) - base_offset_a, 0,
            ((a->stop_times_.back().arr_.time_ / 1440) !=
             (b->stop_times_.front().dep_.time_ / 1440))));
      }
      rule_services.emplace_back(
          CreateRuleService(fbb, fbb.CreateVector(rules)));
    }
  }

  auto const dataset_name =
      std::accumulate(begin(feeds), end(feeds), std::string("GTFS"),
                      [&](std::string const& v,
                          std::pair<std::string const, feed> const& feed_pair) {
                        return v + " - " + feed_pair.second.publisher_name_ +
                               " (" + feed_pair.second.version_ + ")";
                      });

  std::vector<std::tuple<stop const*, stop const*, transfer>>
      missing_symmetry_transfers;
  for (auto const& [p, t] : transfers) {
    auto const& [from, to] = p;
    auto const inverse_it = transfers.find({to, from});
    if (inverse_it == end(transfers)) {
      l(logging::warn, "adding symmetric transfer: {}({}) -> {}({}): {} min",
        to->name_, to->id_, from->name_, from->id_, t.minutes_);
      missing_symmetry_transfers.emplace_back(to, from, t);
    }
  }
  for (auto const& [from, to, t] : missing_symmetry_transfers) {
    transfers.emplace(stop_pair{from, to}, t);
  }

  auto footpaths =
      utl::all(transfers)  //
      | utl::remove_if([](std::pair<stop_pair, transfer>&& t) {
          return t.second.type_ == transfer::NOT_POSSIBLE ||
                 t.first.first == t.first.second;
        })  //
      | utl::transform([&](std::pair<stop_pair, transfer>&& t) {
          return CreateFootpath(fbb, get_or_create_stop(t.first.first),
                                get_or_create_stop(t.first.second),
                                t.second.minutes_);
        })  //
      | utl::vec();

  auto const stop_vec =
      utl::to_vec(stops, [](auto const& s) { return s.second.get(); });
  auto const stop_rtree =
      geo::make_point_rtree(stop_vec, [](auto const& s) { return s->coord_; });
  auto const generate_transfer = [&](stop_pair const& stops) {
    if (stops.first != stops.second &&
        transfers.find(stops) == end(transfers)) {
      footpaths.emplace_back(
          CreateFootpath(fbb, get_or_create_stop(stops.first),
                         get_or_create_stop(stops.second), 2));
      transfers.emplace(stops, transfer{2, transfer::GENERATED});
    }
  };

  if (opt.link_stop_distance_ != 0U) {
    utl::parallel_for(stops, [&](auto const& s) {
      s.second->compute_close_stations(stop_rtree, opt.link_stop_distance_);
    });
  }

  auto const meta_stations =
      utl::all(stops)  //
      | utl::transform([&](auto const& s) {
          return std::make_pair(s.second.get(), s.second->get_metas(stop_vec));
        })  //
      | utl::remove_if([](auto const& s) { return s.second.empty(); })  //
      |
      utl::transform([&](auto const& s_metas) {
        auto const& [this_stop, metas] = s_metas;
        return CreateMetaStation(fbb, get_or_create_stop(this_stop),
                                 fbb.CreateVector(utl::to_vec(
                                     metas, [&, s = this_stop](auto const* eq) {
                                       generate_transfer(std::make_pair(s, eq));
                                       generate_transfer(std::make_pair(eq, s));
                                       return get_or_create_stop(eq);
                                     })));
      })  //
      | utl::vec();

  fbb.Finish(CreateSchedule(fbb, fbb.CreateVector(output_services),
                            fbb.CreateVector(values(fbs_stations)),
                            fbb.CreateVector(values(fbs_routes)), &interval,
                            fbb.CreateVector(footpaths),
                            fbb.CreateVector(rule_services),
                            fbb.CreateVector(meta_stations),
                            fbb.CreateString(dataset_name), hash(root)));

  LOG(logging::info) << "wrote " << n_services << " services";
}

}  // namespace motis::loader::gtfs

// NOLINTEND(bugprone-unchecked-optional-access)
