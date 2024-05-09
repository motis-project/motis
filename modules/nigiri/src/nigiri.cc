#include "motis/nigiri/nigiri.h"

#include <fstream>
#include <utility>

#include "boost/filesystem.hpp"

#include "fmt/std.h"

#include "cista/memory_holder.h"

#include "conf/date_time.h"

#include "utl/enumerate.h"
#include "utl/helpers/algorithm.h"
#include "utl/verify.h"

#include "geo/point_rtree.h"

#include "nigiri/loader/dir.h"
#include "nigiri/loader/gtfs/loader.h"
#include "nigiri/loader/hrd/loader.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/gtfsrt_update.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/rt/util.h"
#include "nigiri/timetable.h"

#include "motis/core/common/logging.h"
#include "motis/module/event_collector.h"
#include "motis/nigiri/geo_station_lookup.h"
#include "motis/nigiri/get_station.h"
#include "motis/nigiri/gtfsrt.h"
#include "motis/nigiri/guesser.h"
#include "motis/nigiri/initial_permalink.h"
#include "motis/nigiri/railviz.h"
#include "motis/nigiri/routing.h"
#include "motis/nigiri/station_lookup.h"
#include "motis/nigiri/trip_to_connection.h"
#include "motis/nigiri/unixtime_conv.h"
#include "utl/parser/split.h"

namespace fbs = flatbuffers;
namespace fs = std::filesystem;
namespace mm = motis::module;
namespace n = nigiri;

namespace motis::nigiri {

struct schedule_info {
  schedule_info(std::string tag, cista::hash_t const hash, fs::path const& path)
      : tag_{std::move(tag)}, sha1sum_{hash}, created_{get_created(path)} {}

  static std::time_t get_created(fs::path const& p) {
    try {
      return boost::filesystem::creation_time(p.string());
    } catch (std::exception const& e) {
      LOG(logging::error) << "boost::filesystem::creation_time:  " << e.what();
      return 0U;
    }
  }

  fbs::Offset<motis::lookup::LookupSchedule> to_fbs(
      fbs::FlatBufferBuilder& fbb) const {
    return lookup::CreateLookupSchedule(fbb, fbb.CreateString(tag_), sha1sum_,
                                        created_);
  }

  std::string tag_;
  cista::hash_t sha1sum_;
  std::time_t created_;
};

struct nigiri::impl {
  impl() {
    loaders_.emplace_back(std::make_unique<n::loader::gtfs::gtfs_loader>());
    loaders_.emplace_back(
        std::make_unique<n::loader::hrd::hrd_5_00_8_loader>());
    loaders_.emplace_back(
        std::make_unique<n::loader::hrd::hrd_5_20_26_loader>());
    loaders_.emplace_back(
        std::make_unique<n::loader::hrd::hrd_5_20_39_loader>());
    loaders_.emplace_back(
        std::make_unique<n::loader::hrd::hrd_5_20_avv_loader>());
  }

  void update_rtt(std::shared_ptr<n::rt_timetable> rtt) {
#if __cpp_lib_atomic_shared_ptr  // not yet supported on macos
    rtt_.store(std::move(rtt));
#else
    auto lock = std::lock_guard{mutex_};
    rtt_ = std::move(rtt);
#endif
  }

  std::shared_ptr<n::rt_timetable> get_rtt() {
#if __cpp_lib_atomic_shared_ptr  // not yet supported on macos
    return rtt_.load();
#else
    std::shared_ptr<n::rt_timetable> copy;
    {
      auto const lock = std::lock_guard{mutex_};
      copy = rtt_;
    }
    return copy;
#endif
  }

  std::vector<std::unique_ptr<n::loader::loader_interface>> loaders_{};
  std::shared_ptr<cista::wrapped<n::timetable>> tt_;
#if __cpp_lib_atomic_shared_ptr  // not yet supported on macos
  std::atomic<std::shared_ptr<n::rt_timetable>> rtt_;
#else
  std::shared_ptr<n::rt_timetable> rtt_;
  std::mutex mutex_;
#endif
  tag_lookup tags_;
  std::shared_ptr<station_lookup> station_lookup_;
  std::vector<gtfsrt> gtfsrt_{};
  std::unique_ptr<guesser> guesser_{};
  std::unique_ptr<railviz> railviz_{};
  std::string initial_permalink_;
  std::vector<schedule_info> schedules_{};
  cista::hash_t hash_{0U};
};

nigiri::nigiri() : module("Next Generation Routing", "nigiri") {
  param(no_cache_, "no_cache", "disable timetable caching");
  param(adjust_footpaths_, "adjust_footpaths",
        "adjust footpaths if they are too fast for the distance");
  param(merge_duplicates_, "match_duplicates",
        "match and merge duplicate trips");
  param(max_footpath_length_, "max_footpath_length",
        "maximum footpath length in minutes");
  param(first_day_, "first_day",
        "YYYY-MM-DD, leave empty to use first day in source data");
  param(num_days_, "num_days", "number of days, ignored if first_day is empty");
  param(lookup_, "lookup", "provide geo station lookup");
  param(guesser_, "guesser", "station typeahead/autocomplete");
  param(railviz_, "railviz", "provide railviz functions");
  param(routing_, "routing", "provide trip_to_connection");
  param(link_stop_distance_, "link_stop_distance",
        "GTFS only: radius to connect stations, 0=skip");
  param(default_timezone_, "default_timezone",
        "tz for agencies w/o tz or routes w/o agency");
  param(gtfsrt_urls_, "gtfsrt",
        "list of GTFS-RT endpoints, format: tag|url|authorization");
  param(gtfsrt_paths_, "gtfsrt_paths",
        "list of GTFS-RT, format: tag|/path/to/file.pb");
  param(gtfsrt_incremental_, "gtfsrt_incremental",
        "true=incremental updates, false=forget all prev. RT updates");
  param(debug_, "debug", "write protobuf JSON files for debugging");
}

nigiri::~nigiri() = default;

void nigiri::init(motis::module::registry& reg) {
  if (!gtfsrt_paths_.empty()) {
    auto const rtt_copy = std::make_shared<n::rt_timetable>(*impl_->get_rtt());
    auto statistics = std::vector<n::rt::statistics>{};
    for (auto const& p : gtfsrt_paths_) {
      auto const [tag, path] = utl::split<'|', utl::cstr, utl::cstr>(p);
      if (path.empty()) {
        throw utl::fail("bad GTFS-RT path: {} (required: tag|path/to/file)", p);
      }
      auto const src = impl_->tags_.get_src(tag.to_str() + '_');
      if (src == n::source_idx_t::invalid()) {
        throw utl::fail("bad GTFS-RT path: tag {} not found", tag.view());
      }
      auto const file =
          cista::mmap{path.c_str(), cista::mmap::protection::READ};
      auto stats = n::rt::statistics{};
      try {
        stats = n::rt::gtfsrt_update_buf(**impl_->tt_, *rtt_copy, src,
                                         tag.view(), file.view());
      } catch (std::exception const& e) {
        stats.parser_error_ = true;
        LOG(logging::error)
            << "GTFS-RT update error (tag=" << tag.view() << ") " << e.what();
      } catch (...) {
        stats.parser_error_ = true;
        LOG(logging::error)
            << "Unknown GTFS-RT update error (tag= " << tag.view() << ")";
      }
      statistics.emplace_back(stats);
    }
    impl_->update_rtt(rtt_copy);
    impl_->railviz_->update(rtt_copy);
    for (auto const [path, stats] : utl::zip(gtfsrt_paths_, statistics)) {
      LOG(logging::info) << "init " << path << ": "
                         << stats.total_entities_success_ << "/"
                         << stats.total_entities_ << " ("
                         << static_cast<double>(stats.total_entities_success_) /
                                stats.total_entities_ * 100
                         << "%)";
    }
  }

  reg.register_op("/nigiri",
                  [&](mm::msg_ptr const& msg) {
                    return route(impl_->tags_, **impl_->tt_,
                                 impl_->get_rtt().get(), msg);
                  },
                  {});

  if (!impl_->tt_->get()->profiles_.empty()) {
    for (auto const& [prf_name, prf_idx] : impl_->tt_->get()->profiles_) {
      reg.register_op(fmt::format("/nigiri/{}", prf_name),
                      [&, p = prf_idx, this](mm::msg_ptr const& msg) {
                        return route(impl_->tags_, **impl_->tt_,
                                     impl_->get_rtt().get(), msg, p);
                      },
                      {});
    }
  }

  if (lookup_) {
    reg.register_op("/lookup/geo_station",
                    [&](mm::msg_ptr const& msg) {
                      return geo_station_lookup(*impl_->station_lookup_, msg);
                    },
                    {});
    reg.register_op("/lookup/station_location",
                    [&](mm::msg_ptr const& msg) {
                      return station_location(impl_->tags_, **impl_->tt_, msg);
                    },
                    {});
    reg.register_op("/lookup/schedule_info",
                    [&](mm::msg_ptr const&) {
                      auto const& tt = (**impl_->tt_);
                      mm::message_creator b;
                      b.create_and_finish(
                          MsgContent_LookupScheduleInfoResponse,
                          lookup::CreateLookupScheduleInfoResponse(
                              b, b.CreateString(fmt::to_string(impl_->hash_)),
                              to_motis_unixtime(tt.external_interval().from_),
                              to_motis_unixtime(tt.external_interval().to_),
                              b.CreateVector(utl::to_vec(
                                  impl_->schedules_,
                                  [&](auto const& s) { return s.to_fbs(b); })))
                              .Union());
                      return make_msg(b);
                    },
                    {});
  }

  if (guesser_) {
    reg.register_op(
        "/guesser",
        [&](mm::msg_ptr const& msg) { return impl_->guesser_->guess(msg); },
        {});
  }

  if (railviz_) {
    reg.register_op("/railviz/map_config",
                    [this](mm::msg_ptr const&) {
                      mm::message_creator mc;
                      mc.create_and_finish(
                          MsgContent_RailVizMapConfigResponse,
                          motis::railviz::CreateRailVizMapConfigResponse(
                              mc, mc.CreateString(impl_->initial_permalink_),
                              mc.CreateString(""))
                              .Union());
                      return make_msg(mc);
                    },
                    {});
    reg.register_op("/railviz/get_trains",
                    [&](mm::msg_ptr const& msg) {
                      return impl_->railviz_->get_trains(msg);
                    },
                    {});
    reg.register_op(
        "/railviz/get_trips",
        [&](mm::msg_ptr const& msg) { return impl_->railviz_->get_trips(msg); },
        {});
    reg.register_op("/railviz/get_station",
                    [&](mm::msg_ptr const& msg) {
                      return get_station(impl_->tags_, **impl_->tt_,
                                         impl_->get_rtt().get(), msg);
                    },
                    {});
  }

  if (routing_) {
    reg.register_op("/trip_to_connection",
                    [&](mm::msg_ptr const& msg) {
                      return trip_to_connection(impl_->tags_, **impl_->tt_,
                                                impl_->get_rtt().get(), msg);
                    },
                    {});
  }

  reg.subscribe("/init", [&]() { register_gtfsrt_timer(*shared_data_); }, {});
}

void nigiri::register_gtfsrt_timer(mm::dispatcher& d) {
  if (!gtfsrt_urls_.empty()) {
    impl_->gtfsrt_ = utl::to_vec(gtfsrt_urls_, [&](auto&& config) {
      return gtfsrt{impl_->tags_, config};
    });
    d.register_timer("RIS GTFS-RT Update",
                     boost::posix_time::seconds{gtfsrt_update_interval_sec_},
                     [&]() { update_gtfsrt(); }, {});
    update_gtfsrt();
  }
}

void nigiri::update_gtfsrt() {
  LOG(logging::info) << "Starting GTFS-RT update: fetch URLs";

  auto const futures = utl::to_vec(
      impl_->gtfsrt_, [](auto& endpoint) { return endpoint.fetch(); });
  auto const today = std::chrono::time_point_cast<date::days>(
      std::chrono::system_clock::now());
  auto const rtt = gtfsrt_incremental_
                       ? std::make_shared<n::rt_timetable>(
                             n::rt_timetable{*impl_->get_rtt()})
                       : std::make_shared<n::rt_timetable>(
                             n::rt::create_rt_timetable(**impl_->tt_, today));
  auto statistics = std::vector<n::rt::statistics>{};
  for (auto const [f, endpoint] : utl::zip(futures, impl_->gtfsrt_)) {
    auto const tag = impl_->tags_.get_tag_clean(endpoint.src());
    auto stats = n::rt::statistics{};
    try {
      auto const& body = f->val().body;
      if (debug_) {
        std::ofstream{fmt::format("{}/{}.json", get_data_directory(), tag)}
            << n::rt::protobuf_to_json(body);
      }
      stats = n::rt::gtfsrt_update_buf(**impl_->tt_, *rtt, endpoint.src(), tag,
                                       body);
    } catch (std::exception const& e) {
      stats.parser_error_ = true;
      LOG(logging::error) << "GTFS-RT update error (tag=" << tag << ") "
                          << e.what();
    } catch (...) {
      stats.parser_error_ = true;
      LOG(logging::error) << "Unknown GTFS-RT update error (tag= " << tag
                          << ")";
    }
    statistics.emplace_back(stats);
  }
  impl_->update_rtt(rtt);
  impl_->railviz_->update(rtt);

  for (auto const [endpoint, stats] : utl::zip(impl_->gtfsrt_, statistics)) {
    LOG(logging::info) << impl_->tags_.get_tag_clean(endpoint.src()) << ": "
                       << stats;
  }
}

void nigiri::import(motis::module::import_dispatcher& reg) {
  impl_ = std::make_unique<impl>();
  std::make_shared<mm::event_collector>(
      get_data_directory().generic_string(), "nigiri", reg,
      [this](mm::event_collector::dependencies_map_t const& dependencies,
             mm::event_collector::publish_fn_t const& publish) {
        using import::FileEvent;

        auto const& msg = dependencies.at("SCHEDULE");

        utl::verify(
            utl::all_of(*motis_content(FileEvent, msg)->paths(),
                        [](auto&& p) {
                          return p->tag()->str() != "schedule" ||
                                 !p->options()->str().empty();
                        }),
            "all schedules require a name tag, even with only one schedule");

        date::sys_days begin;
        auto const today = std::chrono::time_point_cast<date::days>(
            std::chrono::system_clock::now());
        if (first_day_ == "TODAY") {
          begin = today;
        } else {
          std::stringstream ss;
          ss << first_day_;
          ss >> date::parse("%F", begin);
        }

        auto const interval = n::interval<date::sys_days>{
            begin, begin + std::chrono::days{num_days_}};
        LOG(logging::info) << "interval: " << interval.from_ << " - "
                           << interval.to_;

        auto h =
            cista::hash_combine(cista::BASE_HASH,
                                interval.from_.time_since_epoch().count(),  //
                                interval.to_.time_since_epoch().count(),  //
                                adjust_footpaths_, link_stop_distance_,
                                cista::hash(default_timezone_));

        auto datasets =
            std::vector<std::tuple<n::source_idx_t,
                                   decltype(impl_->loaders_)::const_iterator,
                                   std::unique_ptr<n::loader::dir>>>{};
        auto i = 0U;
        for (auto const p : *motis_content(FileEvent, msg)->paths()) {
          if (p->tag()->str() != "schedule") {
            continue;
          }
          auto const path = fs::path{p->path()->str()};
          auto d = n::loader::make_dir(path);
          auto const c = utl::find_if(
              impl_->loaders_, [&](auto&& c) { return c->applicable(*d); });
          utl::verify(c != end(impl_->loaders_), "no loader applicable to {}",
                      path);
          auto const hash = (*c)->hash(*d);
          h = cista::hash_combine(h, hash);

          auto const src = n::source_idx_t{i++};
          datasets.emplace_back(src, c, std::move(d));
          impl_->tags_.add(src, p->options()->str() + "_");

          impl_->schedules_.emplace_back(p->options()->str(), hash, path);
        }
        utl::verify(!datasets.empty(), "no schedule datasets found");

        auto const data_dir = get_data_directory() / "nigiri";
        auto const dump_file_path = data_dir / fmt::to_string(h);

        auto loaded = false;
        for (auto i = 0U; i != 2; ++i) {
          // Parse from input files and write memory image.
          if (no_cache_ || !fs::is_regular_file(dump_file_path)) {
            impl_->tt_ = std::make_shared<cista::wrapped<n::timetable>>(
                cista::raw::make_unique<n::timetable>());

            (*impl_->tt_)->date_range_ = interval;
            n::loader::register_special_stations(**impl_->tt_);

            for (auto const& [src, loader, dir] : datasets) {
              auto progress_tracker = utl::activate_progress_tracker(
                  fmt::format("{}nigiri", impl_->tags_.get_tag(src)));

              LOG(logging::info)
                  << "loading nigiri timetable with configuration "
                  << (*loader)->name();

              try {
                (*loader)->load({.link_stop_distance_ = link_stop_distance_,
                                 .default_tz_ = default_timezone_},
                                src, *dir, **impl_->tt_);
                progress_tracker->status("FINISHED").show_progress(false);
              } catch (std::exception const& e) {
                progress_tracker->status(fmt::format("ERROR: {}", e.what()))
                    .show_progress(false);
                throw;
              } catch (...) {
                progress_tracker->status("ERROR: UNKNOWN EXCEPTION")
                    .show_progress(false);
                throw;
              }
            }

            n::loader::finalize(**impl_->tt_, adjust_footpaths_,
                                merge_duplicates_, max_footpath_length_);

            if (no_cache_) {
              loaded = true;
              break;
            } else {
              // Write to disk, next step: read from disk.
              std::filesystem::create_directories(data_dir);
              (*impl_->tt_)->write(dump_file_path);
            }
          }

          // Read memory image from disk.
          impl_->hash_ = h;
          if (!no_cache_) {
            try {
              impl_->tt_ = std::make_shared<cista::wrapped<n::timetable>>(
                  n::timetable::read(cista::memory_holder{
                      cista::file{dump_file_path.string().c_str(), "r"}
                          .content()}));
              (**impl_->tt_).locations_.resolve_timezones();
              if (!gtfsrt_urls_.empty() || !gtfsrt_paths_.empty()) {
                impl_->update_rtt(std::make_shared<n::rt_timetable>(
                    n::rt::create_rt_timetable(**impl_->tt_, today)));
              }
              loaded = true;
              break;
            } catch (std::exception const& e) {
              LOG(logging::error)
                  << "cannot read cached timetable image: " << e.what();
              std::filesystem::remove(dump_file_path);
              continue;
            }
          }
        }

        utl::verify(loaded, "loading failed");

        LOG(logging::info) << "nigiri timetable: stations="
                           << (*impl_->tt_)->locations_.names_.size()
                           << ", trips=" << (*impl_->tt_)->trip_debug_.size()
                           << "\n";

        if (lookup_) {
          impl_->station_lookup_ = std::make_shared<nigiri_station_lookup>(
              impl_->tags_, **impl_->tt_);
          auto copy = impl_->station_lookup_;
          add_shared_data(to_res_id(mm::global_res_id::STATION_LOOKUP),
                          std::move(copy));
        }

        if (guesser_) {
          impl_->guesser_ =
              std::make_unique<guesser>(impl_->tags_, (**impl_->tt_));
        }

        if (railviz_) {
          impl_->initial_permalink_ = get_initial_permalink(**impl_->tt_);
          impl_->railviz_ =
              std::make_unique<railviz>(impl_->tags_, (**impl_->tt_));
        }

        add_shared_data(to_res_id(mm::global_res_id::NIGIRI_TIMETABLE),
                        impl_->tt_->get());
        add_shared_data(to_res_id(mm::global_res_id::NIGIRI_TAGS),
                        &impl_->tags_);

        import_successful_ = true;
        {
          mm::message_creator fbb;
          fbb.create_and_finish(
              MsgContent_NigiriEvent,
              motis::import::CreateNigiriEvent(fbb, h).Union(), "/import",
              DestinationType_Topic);
          publish(make_msg(fbb));
        }
        {
          mm::message_creator fbb;
          fbb.create_and_finish(MsgContent_StationsEvent,
                                motis::import::CreateStationsEvent(fbb).Union(),
                                "/import", DestinationType_Topic);
          publish(make_msg(fbb));
        }
      })
      ->require("SCHEDULE", [this](mm::msg_ptr const& msg) {
        if (msg->get()->content_type() != MsgContent_FileEvent) {
          return false;
        }
        using import::FileEvent;
        return motis_content(FileEvent, msg)->paths()->size() != 0U &&
               utl::all_of(*motis_content(FileEvent, msg)->paths(),
                           [this](import::ImportPath const* p) {
                             if (p->tag()->str() != "schedule") {
                               return true;
                             }
                             auto const d = n::loader::make_dir(
                                 fs::path{p->path()->str()});
                             return utl::any_of(impl_->loaders_, [&](auto&& c) {
                               return c->applicable(*d);
                             });
                           });
      });
}

}  // namespace motis::nigiri
