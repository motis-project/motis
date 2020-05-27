#include "motis/loader/gtfs/gtfs_parser.h"

#include <numeric>

#include "boost/date_time/posix_time/posix_time.hpp"
#include "boost/filesystem.hpp"

#include "utl/get_or_create.h"
#include "utl/pairwise.h"
#include "utl/pipes/accumulate.h"
#include "utl/pipes/all.h"
#include "utl/pipes/remove_if.h"
#include "utl/pipes/transform.h"
#include "utl/pipes/vec.h"

#include "cista/hash.h"
#include "cista/mmap.h"

#include "geo/latlng.h"

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

using namespace flatbuffers64;
namespace fs = boost::filesystem;
using namespace motis::logging;
using std::get;

namespace motis::loader::gtfs {

auto const required_files = {AGENCY_FILE, STOPS_FILE,      ROUTES_FILE,
                             TRIPS_FILE,  STOP_TIMES_FILE, TRANSFERS_FILE};

cista::hash_t hash(fs::path const& path) {
  auto hash = cista::BASE_HASH;
  auto const hash_file = [&](fs::path const& p) {
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

std::time_t to_unix_time(boost::gregorian::date const& date) {
  boost::posix_time::ptime epoch(boost::gregorian::date(1970, 1, 1));
  return (boost::posix_time::ptime(date) - epoch).total_seconds();
}

void fix_stop_positions(trip_map& trips) {
  auto const is_logical = [](stop_time const& a, stop_time const& b) {
    constexpr auto const SLACK_BUFFER = 5;
    auto const dist_in_m = geo::distance(a.stop_->coord_, b.stop_->coord_);
    if (dist_in_m <= 1.0) {
      return true;
    }
    auto const time_diff_in_min = b.arr_.time_ - a.dep_.time_ + SLACK_BUFFER;
    auto const speed_in_kmh = (dist_in_m / 1000.0) / (time_diff_in_min / 60.0);
    return speed_in_kmh < 350;
  };

  for (auto const& [id, t] : trips) {
    auto const stops = t->stop_times_.to_vector();
    for (auto const [a, b, c] : utl::nwise<3>(stops)) {
      auto const logical_a_b = is_logical(a, b);
      auto const logical_b_c = is_logical(b, c);

      stop* corrected = nullptr;
      geo::latlng before;

      if ((b.stop_->coord_.lat_ < 0.1 && b.stop_->coord_.lng_ < 0.1) ||
          (!logical_a_b && !logical_b_c)) {
        corrected = b.stop_;
        before = b.stop_->coord_;
        b.stop_->coord_ = geo::latlng{
            a.stop_->coord_.lat_ +
                0.5 * (c.stop_->coord_.lat_ - a.stop_->coord_.lat_),
            a.stop_->coord_.lng_ +
                0.5 * (c.stop_->coord_.lng_ - a.stop_->coord_.lng_)};
      } else if (!logical_a_b && logical_b_c) {
        corrected = a.stop_;
        before = a.stop_->coord_;
        a.stop_->coord_ = b.stop_->coord_;
      } else if (logical_a_b && !logical_b_c) {
        corrected = c.stop_;
        before = c.stop_->coord_;
        c.stop_->coord_ = b.stop_->coord_;
      }

      if (corrected != nullptr) {
        LOG(warn) << "adjusting stop position from (" << before.lat_ << ", "
                  << before.lng_ << ") to (" << corrected->coord_.lat_ << ", "
                  << corrected->coord_.lng_
                  << "): sanity check failed for trip \"" << id
                  << "\", stop_id=" << corrected->id_ << " (a=" << a.stop_->id_
                  << ", b=" << b.stop_->id_ << ", c=" << c.stop_->id_
                  << "): " << a.stop_->coord_.lat_ << ","
                  << a.stop_->coord_.lng_ << " -> " << b.stop_->coord_.lat_
                  << "," << b.stop_->coord_.lng_ << " -> "
                  << c.stop_->coord_.lat_ << "," << c.stop_->coord_.lng_;
      }
    }
  }
}

void gtfs_parser::parse(fs::path const& root, FlatBufferBuilder& fbb) {
  motis::logging::scoped_timer global_timer{"gtfs parser"};

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
  auto const services = traffic_days(calendar, dates);
  auto transfers = read_transfers(load(TRANSFERS_FILE), stops);
  auto trips = read_trips(load(TRIPS_FILE), routes, services);
  read_stop_times(load(STOP_TIMES_FILE), trips, stops);
  fix_stop_positions(trips);

  std::map<category, Offset<Category>> fbs_categories;
  std::map<agency const*, Offset<Provider>> fbs_providers;
  std::map<std::string, Offset<String>> fbs_strings;
  std::map<stop const*, Offset<Station>> fbs_stations;
  std::map<trip::stop_seq, Offset<Route>> fbs_routes;
  std::vector<Offset<Service>> fbs_services;

  auto get_or_create_stop = [&](stop const* s) {
    return utl::get_or_create(fbs_stations, s, [&]() {
      return CreateStation(
          fbb, fbb.CreateString(s->id_), fbb.CreateString(s->name_),
          s->coord_.lat_, s->coord_.lng_, 2,
          fbb.CreateVector(std::vector<Offset<String>>()), 0,
          s->timezone_.empty() ? 0 : fbb.CreateString(s->timezone_));
    });
  };

  auto get_or_create_category = [&](route const* r) {
    if (auto const cat = r->get_category(); cat.has_value()) {
      return utl::get_or_create(fbs_categories, *cat, [&]() {
        return CreateCategory(fbb, fbb.CreateString(cat->name_),
                              cat->output_rule_);
      });
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
      return get_or_create_str(std::get<0>(t->stops().back())->name_);
    }
  };

  motis::logging::scoped_timer export_timer{"export"};
  motis::logging::clog_import_step("Export schedule.raw", 40, 80, trips.size());
  auto const interval =
      Interval{static_cast<uint64_t>(to_unix_time(services.first_day_)),
               static_cast<uint64_t>(to_unix_time(services.last_day_))};

  auto i = size_t{0U};
  auto const output_services = fbb.CreateVector(
      utl::all(trips)  //
      | utl::remove_if([](auto const& trp) {
          auto const stop_count = trp.second->stops().size();
          if (stop_count < 2) {
            LOG(warn) << "invalid trip " << trp.first << ": "
                      << trp.second->stops().size() << " stops";
          }
          return stop_count < 2;
        })  //
      |
      utl::transform([&](auto const& entry) {
        motis::logging::clog_import_progress(i++, 100000);

        auto const& t = entry.second;
        auto const stop_seq = t->stops();
        return CreateService(
            fbb,
            utl::get_or_create(
                fbs_routes, stop_seq,
                [&]() {
                  return CreateRoute(
                      fbb,  //
                      fbb.CreateVector(utl::to_vec(
                          begin(stop_seq), end(stop_seq),
                          [&](trip::stop_identity const& s) {
                            return get_or_create_stop(std::get<0>(s));
                          })),
                      fbb.CreateVector(utl::to_vec(
                          begin(stop_seq), end(stop_seq),
                          [](trip::stop_identity const& s) {
                            return static_cast<uint8_t>(std::get<1>(s) ? 1u
                                                                       : 0u);
                          })),
                      fbb.CreateVector(utl::to_vec(
                          begin(stop_seq), end(stop_seq),
                          [](trip::stop_identity const& s) {
                            return static_cast<uint8_t>(std::get<2>(s) ? 1u
                                                                       : 0u);
                          })));
                }),
            fbb.CreateString(serialize_bitset(*t->service_)),
            fbb.CreateVector(repeat_n(
                CreateSection(
                    fbb, get_or_create_category(t->route_),
                    get_or_create_provider(t->route_->agency_),
                    !t->short_name_.empty() &&
                            std::all_of(begin(t->short_name_),
                                        end(t->short_name_),
                                        [](auto&& c) -> bool {
                                          return std::isdigit(c);
                                        })
                        ? std::stoi(t->short_name_)
                        : 0,
                    get_or_create_str(t->route_->short_name_),
                    fbb.CreateVector(std::vector<Offset<Attribute>>()),
                    CreateDirection(fbb, 0, get_or_create_direction(t.get()))),
                stop_seq.size() - 1)),
            0,
            fbb.CreateVector(utl::all(t->stop_times_)  //
                             | utl::accumulate(
                                   [&](std::vector<int>&& times,
                                       flat_map<stop_time>::entry_t const& st) {
                                     times.emplace_back(st.second.arr_.time_);
                                     times.emplace_back(st.second.dep_.time_);
                                     return std::move(times);
                                   },
                                   std::vector<int>())),
            0,
            CreateServiceDebugInfo(fbb, fbb.CreateString(""),
                                   entry.second->line_, entry.second->line_),
            false, 0, get_or_create_str(entry.first));  // Trip ID
      }) |
      utl::vec());

  auto const dataset_name =
      std::accumulate(begin(feeds), end(feeds), std::string("GTFS"),
                      [&](std::string const& v,
                          std::pair<std::string const, feed> const& feed_pair) {
                        return v + " - " + feed_pair.second.publisher_name_ +
                               " (" + feed_pair.second.version_ + ")";
                      });

  auto footpaths =
      utl::all(transfers)  //
      | utl::remove_if([](std::pair<stop_pair, transfer>&& t) {
          return t.second.type_ != transfer::TIMED_TRANSFER;
        })  //
      | utl::transform([&](std::pair<stop_pair, transfer>&& t) {
          return CreateFootpath(fbb, get_or_create_stop(t.first.first),
                                get_or_create_stop(t.first.second),
                                t.second.minutes_);
        })  //
      | utl::vec();

  auto const generate_transfer = [&](stop_pair const& stops) {
    if (transfers.find(stops) == end(transfers)) {
      footpaths.emplace_back(
          CreateFootpath(fbb, get_or_create_stop(stops.first),
                         get_or_create_stop(stops.second), 2));
      transfers.emplace(stops, transfer{2, transfer::GENERATED});
    }
  };
  auto const meta_stations =
      utl::all(stops)  //
      | utl::remove_if(
            [](auto const& s) { return s.second->get_metas().empty(); })  //
      | utl::transform([&](auto const& s) {
          stop* this_stop = s.second.get();
          return CreateMetaStation(
              fbb, get_or_create_stop(this_stop),
              fbb.CreateVector(
                  utl::to_vec(this_stop->get_metas(), [&](auto const* eq) {
                    generate_transfer(std::make_pair(this_stop, eq));
                    generate_transfer(std::make_pair(eq, this_stop));
                    return get_or_create_stop(eq);
                  })));
        })  //
      | utl::vec();

  fbb.Finish(CreateSchedule(
      fbb, output_services, fbb.CreateVector(values(fbs_stations)),
      fbb.CreateVector(values(fbs_routes)), &interval,
      fbb.CreateVector(footpaths),
      fbb.CreateVector(std::vector<Offset<RuleService>>()),
      fbb.CreateVector(meta_stations), fbb.CreateString(dataset_name),
      hash(root)));
}

}  // namespace motis::loader::gtfs
