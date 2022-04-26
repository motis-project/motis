#include "motis/gbfs/gbfs.h"

#include <mutex>
#include <numeric>

#include "utl/concat.h"
#include "utl/enumerate.h"
#include "utl/erase_duplicates.h"
#include "utl/pipes.h"

#include "geo/point_rtree.h"

#include "motis/core/common/logging.h"
#include "motis/core/schedule/schedule.h"
#include "motis/core/conv/position_conv.h"
#include "motis/core/conv/station_conv.h"
#include "motis/module/context/motis_call.h"
#include "motis/module/context/motis_http_req.h"
#include "motis/module/context/motis_parallel_for.h"
#include "motis/module/event_collector.h"
#include "motis/module/message.h"
#include "motis/gbfs/free_bike.h"
#include "motis/gbfs/station.h"
#include "motis/gbfs/system_information.h"
#include "motis/gbfs/system_status.h"

namespace fbs = flatbuffers;
using namespace motis::logging;
using namespace motis::module;

namespace motis::gbfs {

struct positions {};

struct gbfs::impl {
  explicit impl(config const& c, schedule const& sched)
      : config_{c}, sched_{sched} {}

  void fetch_stream(std::string url) {
    auto tag = std::string{};
    auto vehicle_type = std::string{"bike"};
    auto const tag_pos = url.find('|');
    if (tag_pos != std::string::npos) {
      tag = url.substr(0, tag_pos);

      auto const vehicle_type_delimiter = tag.find('-');
      if (vehicle_type_delimiter != std::string::npos) {
        vehicle_type = tag.substr(vehicle_type_delimiter + 1);
        tag = tag.substr(0, vehicle_type_delimiter);
      }

      url = url.substr(tag_pos + 1);
    }

    auto const s = read_system_status(motis_http(url)->val().body);
    if (s.empty()) {
      l(warn, "no feeds from {}", url);
      return;
    }

    auto const& urls = s.front();

    auto f_station_info = http_future_t{};
    auto f_station_status = http_future_t{};
    auto f_free_bikes = http_future_t{};
    auto f_system_info = http_future_t{};

    if (urls.station_info_url_.has_value()) {
      f_station_info = motis_http(*urls.station_info_url_);
      f_station_status = motis_http(*urls.station_status_url_);
    }

    if (urls.free_bike_url_.has_value()) {
      f_free_bikes = motis_http(*urls.free_bike_url_);
    }

    if (urls.system_information_url_.has_value()) {
      f_system_info = motis_http(*urls.system_information_url_);
    }

    auto const lock = std::scoped_lock{mutex_};
    auto& info = status_[tag];
    info.vehicle_type_ = vehicle_type;
    if (urls.station_info_url_.has_value()) {
      info.stations_ =
          utl::to_vec(parse_stations(tag, f_station_info->val().body,
                                     f_station_status->val().body),
                      [](auto const& el) { return el.second; });
      info.stations_rtree_ = geo::make_point_rtree(
          utl::to_vec(info.stations_, [](auto&& s) { return s.pos_; }));
    }
    if (urls.free_bike_url_.has_value()) {
      info.free_bikes_ = parse_free_bikes(tag, f_free_bikes->val().body);
      info.free_bikes_rtree_ = geo::make_point_rtree(
          utl::to_vec(info.free_bikes_, [](auto&& s) { return s.pos_; }));
    }
    if (urls.system_information_url_.has_value()) {
      info.info_ = read_system_information(f_system_info->val().body);
    }
  }

  void init(schedule const& sched) {
    auto const t = scoped_timer{"GBFS init"};
    motis_parallel_for(config_.urls_, [&](auto&& url) { fetch_stream(url); });

    auto const lock = std::scoped_lock{mutex_};
    pt_stations_rtree_ =
        geo::make_point_rtree(sched.stations_, [](auto const& s) {
          return geo::latlng{s->lat(), s->lng()};
        });
    for (auto const& [tag, info] : status_) {
      l(logging::info,
        "GBFS {} (type={}): loaded {} stations, {} free vehicles", tag,
        info.vehicle_type_, info.stations_.size(), info.free_bikes_.size());
    }
  }

  static msg_ptr make_one_to_many(std::string const& profile,
                                  geo::latlng const& one,
                                  std::vector<geo::latlng> const& many,
                                  SearchDir direction) {
    auto const fbs_pos = to_fbs(one);
    message_creator mc;
    mc.create_and_finish(MsgContent_OSRMOneToManyRequest,
                         osrm::CreateOSRMOneToManyRequest(
                             mc, mc.CreateString(profile), direction, &fbs_pos,
                             mc.CreateVectorOfStructs(utl::to_vec(
                                 many, [](auto&& p) { return to_fbs(p); })))
                             .Union(),
                         "/osrm/one_to_many");
    return make_msg(mc);
  }

  static msg_ptr make_table_request(std::string const& profile,
                                    std::vector<geo::latlng> const& from,
                                    std::vector<geo::latlng> const& to) {
    message_creator mc;
    mc.create_and_finish(MsgContent_OSRMManyToManyRequest,
                         osrm::CreateOSRMManyToManyRequest(
                             mc, mc.CreateString(profile),
                             mc.CreateVectorOfStructs(utl::to_vec(
                                 from, [](auto&& p) { return to_fbs(p); })),
                             mc.CreateVectorOfStructs(utl::to_vec(
                                 to, [](auto&& p) { return to_fbs(p); })))
                             .Union(),
                         "/osrm/table");
    return make_msg(mc);
  }

  static msg_ptr empty_response(SearchDir const dir) {
    message_creator fbb;
    fbb.create_and_finish(
        MsgContent_GBFSRoutingResponse,
        CreateGBFSRoutingResponse(
            fbb, dir,
            fbb.CreateVector(std::vector<flatbuffers::Offset<RouteInfo>>{}))
            .Union());
    return make_msg(fbb);
  }

  msg_ptr route(schedule const&, msg_ptr const& m) {
    using osrm::OSRMManyToManyResponse;
    using osrm::OSRMOneToManyResponse;

    struct journey {
      struct invalid {};
      struct s {  // station bound
        duration first_walk_duration_{0};
        duration bike_duration_{0};
        duration second_walk_duration_{0};
        uint32_t sx_, sp_, p_;
      };
      struct b {  // free-float
        duration walk_duration_{0};
        duration bike_duration_{0};
        uint32_t b_, p_;
      };

      bool valid() const { return !std::holds_alternative<invalid>(info_); }

      uint16_t total_duration_{std::numeric_limits<uint16_t>::max()};
      std::variant<invalid, s, b> info_{invalid{}};
    };

    constexpr auto const max_walk_speed = 1.1;  // m/s 4km/h
    constexpr auto const max_bike_speed = 7.0;  // m/s 25km/h
    constexpr auto const max_car_speed = 27.8;  // m/s 100km/h

    auto const req = motis_content(GBFSRoutingRequest, m);

    auto const provider = req->provider()->str();

    auto const lock = std::lock_guard{mutex_};
    auto const& info = status_.at(provider);
    auto const& stations = info.stations_;
    auto const& stations_rtree = info.stations_rtree_;
    auto const& free_bikes = info.free_bikes_;
    auto const& free_bikes_rtree = info.free_bikes_rtree_;
    auto const& vehicle_type = info.vehicle_type_;

    auto const max_walk_duration = req->max_foot_duration();
    auto const max_bike_duration = req->max_bike_duration();
    auto const max_walk_dist = max_walk_duration * 60 * max_walk_speed;
    auto const max_bike_dist =
        req->max_bike_duration() * 60 *
        (vehicle_type == "bike" ? max_bike_speed : max_car_speed);
    auto const max_total_dist = max_walk_dist + max_bike_dist;

    auto const x = from_fbs(req->x());

    auto const p = pt_stations_rtree_.in_radius(x, max_total_dist);
    if (p.empty()) {
      return empty_response(req->dir());
    }

    auto const p_pos = utl::to_vec(p, [&](auto const idx) {
      auto const& s = *sched_.stations_.at(idx);
      return geo::latlng{s.lat(), s.lng()};
    });

    auto const sx = stations_rtree.in_radius(x, max_walk_dist);
    auto const sx_pos =
        utl::to_vec(sx, [&](auto const idx) { return stations.at(idx).pos_; });
    auto sp = std::accumulate(
        begin(p_pos), end(p_pos), std::vector<size_t>{},
        [&](std::vector<size_t> acc, geo::latlng const& pt_station_pos) {
          return utl::concat(
              acc, stations_rtree.in_radius(pt_station_pos, max_walk_dist));
        });
    utl::erase_duplicates(sp);
    auto const sp_pos =
        utl::to_vec(sp, [&](auto const idx) { return stations.at(idx).pos_; });
    auto b = [&]() {
      if (req->dir() == SearchDir_Forward) {
        return free_bikes_rtree.in_radius(x, max_walk_dist);
      } else {
        return std::accumulate(
            begin(p), end(p), std::vector<size_t>{},
            [&](std::vector<size_t> acc, size_t const idx) {
              auto const& s = *sched_.stations_.at(idx);
              acc.emplace_back(free_bikes_rtree.nearest({s.lat(), s.lng()}, 1U)
                                   .at(0)
                                   .second);
              return acc;
            });
      }
    }();
    utl::erase_duplicates(b);

    if (b.empty() && (sp.empty() || sx.empty())) {
      return empty_response(req->dir());
    }

    auto const b_pos =
        utl::to_vec(b, [&](auto const idx) { return free_bikes.at(idx).pos_; });

    auto p_best_journeys = std::vector<journey>{};
    p_best_journeys.resize(p.size());

    if (req->dir() == SearchDir_Forward) {
      // REQUESTS
      // free-float FWD: x --walk--> [b] --bike--> [p]
      auto const f_x_to_b_walks =
          b.empty() ? future{}
                    : motis_call(make_one_to_many("foot", x, b_pos,
                                                  SearchDir_Forward));
      auto const f_b_to_p_rides =
          b.empty()
              ? future{}
              : motis_call(make_table_request(vehicle_type, b_pos, p_pos));

      // REQUESTS
      // station BWD: [p] --walk--> [sp] --bike--> [sx] --walk--> x
      auto const f_x_to_sx_walks =
          (sx.empty() || sp.empty())
              ? future{}
              : motis_call(
                    make_one_to_many("foot", x, sx_pos, SearchDir_Forward));
      auto const f_sx_to_sp_rides =
          (sx.empty() || sp.empty())
              ? future{}
              : motis_call(make_table_request(vehicle_type, sx_pos, sp_pos));
      auto const f_sp_to_p_walks =
          (sx.empty() || sp.empty())
              ? future{}
              : motis_call(make_table_request("foot", sp_pos, p_pos));

      // BUILD JOURNEYS
      // free-float FWD: x --walk--> [b] --bike--> [p]
      if (f_b_to_p_rides) {
        auto const b_to_p_table =
            motis_content(OSRMManyToManyResponse, f_b_to_p_rides->val())
                ->costs();
        for (auto const& [b_vec_idx, x_to_b_res] : utl::enumerate(
                 *motis_content(OSRMOneToManyResponse, f_x_to_b_walks->val())
                      ->costs())) {
          auto const x_to_b_walk_duration =
              static_cast<duration>(x_to_b_res->duration() / 60.0);
          if (x_to_b_walk_duration > max_walk_duration) {
            continue;
          }

          for (auto const& [p_vec_idx, p_id] : utl::enumerate(p)) {
            auto const b_to_p_duration = static_cast<duration>(
                b_to_p_table->Get(b_vec_idx * p.size() + p_vec_idx) / 60.0);
            if (b_to_p_duration > max_bike_duration) {
              continue;
            }

            auto const total_duration = x_to_b_walk_duration + b_to_p_duration;
            if (auto& best = p_best_journeys[p_vec_idx];
                best.total_duration_ > total_duration) {
              best.total_duration_ = total_duration;
              best.info_ = journey::b{x_to_b_walk_duration, b_to_p_duration,
                                      static_cast<uint32_t>(b_vec_idx),
                                      static_cast<uint32_t>(p_vec_idx)};
            }
          }
        }
      }

      // BUILD JOURNEYS
      // station FWD: x --walk--> [sx] --bike--> [sp] --walk--> [p]
      if (f_sx_to_sp_rides) {
        auto const sx_to_sp_table =
            motis_content(OSRMManyToManyResponse, f_sx_to_sp_rides->val())
                ->costs();
        auto const sp_to_p_table =
            motis_content(OSRMManyToManyResponse, f_sp_to_p_walks->val())
                ->costs();
        for (auto const [sx_vec_idx, x_to_sx_res] : utl::enumerate(
                 *motis_content(OSRMOneToManyResponse, f_x_to_sx_walks->val())
                      ->costs())) {
          auto const x_to_sx_walk_duration =
              static_cast<duration>(x_to_sx_res->duration() / 60.0);
          if (x_to_sx_walk_duration > max_walk_duration) {
            continue;
          }

          for (auto const& [sp_vec_idx, sp_id] : utl::enumerate(sp)) {
            auto const sx_to_sp_duration = static_cast<duration>(
                sx_to_sp_table->Get(sx_vec_idx * sp.size() + sp_vec_idx) /
                60.0);
            if (sx_to_sp_duration > max_bike_duration) {
              continue;
            }

            for (auto const& [p_vec_idx, p_id] : utl::enumerate(p)) {
              auto const sp_to_p_walk_duration = static_cast<duration>(
                  sp_to_p_table->Get(sp_vec_idx * p.size() + p_vec_idx) / 60.0);
              if (sp_to_p_walk_duration > max_walk_duration ||
                  x_to_sx_walk_duration + sp_to_p_walk_duration >
                      max_walk_duration) {
                continue;
              }

              auto const total_duration = x_to_sx_walk_duration +
                                          sx_to_sp_duration +
                                          sp_to_p_walk_duration;
              if (auto& best = p_best_journeys[p_vec_idx];
                  best.total_duration_ > total_duration) {
                best.total_duration_ = total_duration;
                best.info_ = journey::s{x_to_sx_walk_duration,
                                        sx_to_sp_duration,
                                        sp_to_p_walk_duration,
                                        static_cast<uint32_t>(sx_vec_idx),
                                        static_cast<uint32_t>(sp_vec_idx),
                                        static_cast<uint32_t>(p_vec_idx)};
              }
            }
          }
        }
      }
    } else {
      // REQUESTS
      // free-float BWD: [p] --walk--> [b] --bike--> x
      auto const f_p_to_b_walks =
          b_pos.empty() ? future{}
                        : motis_call(make_table_request("foot", p_pos, b_pos));
      auto const f_b_to_x_rides =
          b_pos.empty() ? future{}
                        : motis_call(make_one_to_many(vehicle_type, x, b_pos,
                                                      SearchDir_Backward));

      // REQUESTS
      // station BWD: [p] --walk--> [sp] --bike--> [sx] --walk--> x
      auto const f_p_to_sp_walks =
          (sp_pos.empty() || sx_pos.empty())
              ? future{}
              : motis_call(make_table_request("foot", p_pos, sp_pos));
      auto const f_sp_to_sx_rides =
          (sp_pos.empty() || sx_pos.empty())
              ? future{}
              : motis_call(make_table_request(vehicle_type, sp_pos, sx_pos));
      auto const f_sx_to_x_walks =
          (sp_pos.empty() || sx_pos.empty())
              ? future{}
              : motis_call(
                    make_one_to_many("foot", x, sx_pos, SearchDir_Backward));

      // BUILD JOURNEYS
      // free-float BWD: [p] --walk--> [b] --bike--> x
      if (f_p_to_b_walks) {
        auto const p_to_b_table =
            motis_content(OSRMManyToManyResponse, f_p_to_b_walks->val())
                ->costs();
        for (auto const& [b_vec_idx, b_to_x_res] : utl::enumerate(
                 *motis_content(OSRMOneToManyResponse, f_b_to_x_rides->val())
                      ->costs())) {
          auto const b_to_x_bike_duration =
              static_cast<duration>(b_to_x_res->duration() / 60.0);
          if (b_to_x_bike_duration > max_bike_duration) {
            continue;
          }

          for (auto const& [p_vec_idx, p_id] : utl::enumerate(p)) {
            auto const p_to_b_walk_duration = static_cast<duration>(
                p_to_b_table->Get(p_vec_idx * b.size() + b_vec_idx) / 60.0);
            if (p_to_b_walk_duration > max_walk_duration) {
              continue;
            }

            auto const total_duration =
                p_to_b_walk_duration + b_to_x_bike_duration;
            if (auto& best = p_best_journeys[p_vec_idx];
                best.total_duration_ > total_duration) {
              best.total_duration_ = total_duration;
              best.info_ =
                  journey::b{p_to_b_walk_duration, b_to_x_bike_duration,
                             static_cast<uint32_t>(b_vec_idx),
                             static_cast<uint32_t>(p_vec_idx)};
            }
          }
        }
      }

      // BUILD JOURNEYS
      // station BWD: [p] --walk--> [sp] --bike--> [sx] --walk--> x
      if (f_p_to_b_walks && f_sp_to_sx_rides) {
        auto const p_to_sp_table =
            motis_content(OSRMManyToManyResponse, f_p_to_sp_walks->val())
                ->costs();
        auto const sp_to_sx_table =
            motis_content(OSRMManyToManyResponse, f_sp_to_sx_rides->val())
                ->costs();
        for (auto const [sx_vec_idx, sx_to_x_res] : utl::enumerate(
                 *motis_content(OSRMOneToManyResponse, f_sx_to_x_walks->val())
                      ->costs())) {
          auto const sx_to_x_walk_duration =
              static_cast<duration>(sx_to_x_res->duration() / 60.0);
          if (sx_to_x_walk_duration > max_walk_duration) {
            continue;
          }

          for (auto const& [sp_vec_idx, sp_id] : utl::enumerate(sp)) {
            auto const sp_to_sx_duration = static_cast<duration>(
                sp_to_sx_table->Get(sp_vec_idx * sx.size() + sx_vec_idx) /
                60.0);

            if (sp_to_sx_duration > max_bike_duration) {
              continue;
            }

            for (auto const& [p_vec_idx, p_id] : utl::enumerate(p)) {
              auto const p_to_sp_walk_duration = static_cast<duration>(
                  p_to_sp_table->Get(p_vec_idx * sp.size() + sp_vec_idx) /
                  60.0);
              if (p_to_sp_walk_duration > max_walk_duration ||
                  p_to_sp_walk_duration + sx_to_x_walk_duration >
                      max_walk_duration) {
                continue;
              }

              auto const total_duration = p_to_sp_walk_duration +
                                          sp_to_sx_duration +
                                          sx_to_x_walk_duration;
              if (auto& best = p_best_journeys[p_vec_idx];
                  best.total_duration_ > total_duration) {
                best.total_duration_ = total_duration;
                best.info_ = journey::s{p_to_sp_walk_duration,
                                        sp_to_sx_duration,
                                        sx_to_x_walk_duration,
                                        static_cast<uint32_t>(sx_vec_idx),
                                        static_cast<uint32_t>(sp_vec_idx),
                                        static_cast<uint32_t>(p_vec_idx)};
              }
            }
          }
        }
      }
    }

    message_creator fbb;
    auto const routes =
        utl::all(p_best_journeys)  //
        | utl::remove_if([](journey const& j) { return !j.valid(); })  //
        | utl::transform([&](journey const& j) {
            return std::visit(
                utl::overloaded{
                    [&](journey::invalid const&) {
                      throw std::runtime_error{"unreachable"};
                      return fbs::Offset<RouteInfo>{};
                    },

                    [&](journey::b const& free_bike_info) {
                      auto const& free_bike =
                          free_bikes.at(b.at(free_bike_info.b_));
                      auto const& station =
                          *sched_.stations_.at(p.at(free_bike_info.p_));
                      auto const pos = to_fbs(free_bike.pos_);
                      return CreateRouteInfo(
                          fbb, fbb.CreateString(vehicle_type),
                          to_fbs(fbb, station), BikeRoute_FreeBikeRoute,
                          CreateFreeBikeRoute(
                              fbb, fbb.CreateString(free_bike.id_), &pos,
                              free_bike_info.walk_duration_,
                              free_bike_info.bike_duration_)
                              .Union(),
                          j.total_duration_);
                    },

                    [&](journey::s const& station_bike_info) {
                      auto const& sx_bike_station =
                          stations.at(sx.at(station_bike_info.sx_));
                      auto const& sp_bike_station =
                          stations.at(sp.at(station_bike_info.sp_));
                      auto const& station =
                          *sched_.stations_.at(p.at(station_bike_info.p_));
                      auto const sx_pos = to_fbs(sx_bike_station.pos_);
                      auto const sp_pos = to_fbs(sp_bike_station.pos_);
                      auto const sx_gbfs_station = CreateGBFSStation(
                          fbb, fbb.CreateString(sx_bike_station.id_),
                          fbb.CreateString(sx_bike_station.name_), &sx_pos);
                      auto const sp_gbfs_station = CreateGBFSStation(
                          fbb, fbb.CreateString(sp_bike_station.id_),
                          fbb.CreateString(sp_bike_station.name_), &sp_pos);
                      return CreateRouteInfo(
                          fbb, fbb.CreateString(vehicle_type),
                          to_fbs(fbb, station), BikeRoute_StationBikeRoute,
                          CreateStationBikeRoute(
                              fbb,
                              req->dir() == SearchDir_Forward ? sx_gbfs_station
                                                              : sp_gbfs_station,
                              req->dir() == SearchDir_Forward ? sp_gbfs_station
                                                              : sx_gbfs_station,
                              station_bike_info.first_walk_duration_,
                              station_bike_info.bike_duration_,
                              station_bike_info.second_walk_duration_)
                              .Union(),
                          j.total_duration_);
                    },
                },
                j.info_);
          })  //
        | utl::vec();

    fbb.create_and_finish(
        MsgContent_GBFSRoutingResponse,
        CreateGBFSRoutingResponse(fbb, req->dir(), fbb.CreateVector(routes))
            .Union());

    return make_msg(fbb);
  }

  msg_ptr info() const {
    message_creator fbb;
    fbb.create_and_finish(
        MsgContent_GBFSProvidersResponse,
        CreateGBFSProvidersResponse(
            fbb, fbb.CreateVector(utl::to_vec(
                     status_,
                     [&](auto&& s) {
                       auto const& [tag, info] = s;
                       return CreateGBFSProvider(
                           fbb, fbb.CreateString(tag),
                           fbb.CreateString(info.info_.name_),
                           fbb.CreateString(info.info_.name_short_),
                           fbb.CreateString(info.info_.operator_),
                           fbb.CreateString(info.info_.url_),
                           fbb.CreateString(info.info_.purchase_url_),
                           fbb.CreateString(info.info_.mail_),
                           fbb.CreateString(info.vehicle_type_));
                     })))
            .Union());
    return make_msg(fbb);
  }

  struct provider_status {
    system_information info_;
    std::string vehicle_type_;
    std::vector<station> stations_;
    std::vector<free_bike> free_bikes_;
    geo::point_rtree free_bikes_rtree_, stations_rtree_;
  };

  config const& config_;
  schedule const& sched_;
  std::mutex mutex_;
  std::map<std::string, provider_status> status_;
  geo::point_rtree pt_stations_rtree_;
};

gbfs::gbfs() : module("RIS", "gbfs") {
  param(config_.update_interval_minutes_, "update_interval",
        "update interval in minutes");
  param(config_.urls_, "urls", "URLs to fetch data from");
}

gbfs::~gbfs() = default;

void gbfs::import(import_dispatcher& reg) {
  using import::OSRMEvent;
  std::make_shared<event_collector>(
      get_data_directory().generic_string(), "gbfs", reg,
      [this](event_collector::dependencies_map_t const&,
             event_collector::publish_fn_t const&) {
        import_successful_ = true;
      })
      ->require("SCHEDULE",
                [](msg_ptr const& msg) {
                  return msg->get()->content_type() == MsgContent_ScheduleEvent;
                })
      ->require("OSRM_BIKE",
                [](msg_ptr const& msg) {
                  return msg->get()->content_type() == MsgContent_OSRMEvent &&
                         motis_content(OSRMEvent, msg)->profile()->str() ==
                             "bike";
                })
      ->require("OSRM_FOOT",
                [](msg_ptr const& msg) {
                  return msg->get()->content_type() == MsgContent_OSRMEvent &&
                         motis_content(OSRMEvent, msg)->profile()->str() ==
                             "foot";
                })
      ->require("OSRM_CAR", [](msg_ptr const& msg) {
        return msg->get()->content_type() == MsgContent_OSRMEvent &&
               motis_content(OSRMEvent, msg)->profile()->str() == "car";
      });
}

void gbfs::init(motis::module::registry& r) {
  impl_ = std::make_unique<impl>(config_, get_sched());
  r.subscribe("/init", [&]() { impl_->init(get_sched()); });
  r.register_op("/gbfs/route",
                [&](msg_ptr const& m) { return impl_->route(get_sched(), m); });
  r.register_op("/gbfs/info", [&](msg_ptr const&) { return impl_->info(); });
}

}  // namespace motis::gbfs
