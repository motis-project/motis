#include "motis/gbfs/gbfs.h"

#include <mutex>
#include <numeric>

#include "utl/concat.h"
#include "utl/enumerate.h"

#include "geo/point_rtree.h"

#include "motis/core/common/logging.h"
#include "motis/core/schedule/schedule.h"
#include "motis/core/conv/position_conv.h"
#include "motis/module/context/motis_call.h"
#include "motis/module/context/motis_http_req.h"
#include "motis/module/context/motis_parallel_for.h"
#include "motis/module/event_collector.h"
#include "motis/gbfs/free_bike.h"
#include "motis/gbfs/station.h"
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
    auto const tag_pos = url.find('|');
    if (tag_pos != std::string::npos) {
      tag = url.substr(0, tag_pos) + "-";
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

    if (urls.station_info_url_.has_value()) {
      f_station_info = motis_http(*urls.station_info_url_);
      f_station_status = motis_http(*urls.station_status_url_);
    }

    if (urls.free_bike_url_.has_value()) {
      f_free_bikes = motis_http(*urls.free_bike_url_);
    }

    auto const lock = std::scoped_lock{mutex_};
    if (urls.station_info_url_.has_value()) {
      utl::concat(stations_,
                  utl::to_vec(parse_stations(tag, f_station_info->val().body,
                                             f_station_status->val().body),
                              [](auto const& el) { return el.second; }));
    }
    if (urls.free_bike_url_.has_value()) {
      utl::concat(free_bikes_, parse_free_bikes(tag, f_free_bikes->val().body));
    }
  }

  void init(schedule const& sched) {
    auto const t = scoped_timer{"GBFS init"};
    motis_parallel_for(config_.urls_, [&](auto&& url) { fetch_stream(url); });

    auto const lock = std::scoped_lock{mutex_};
    stations_rtree_ = geo::make_point_rtree(
        utl::to_vec(stations_, [](auto&& s) { return s.pos_; }));
    free_bikes_rtree_ = geo::make_point_rtree(
        utl::to_vec(free_bikes_, [](auto&& s) { return s.pos_; }));
    pt_stations_rtree_ =
        geo::make_point_rtree(sched.stations_, [](auto const& s) {
          return geo::latlng{s->lat(), s->lng()};
        });
    l(info, "loaded {} stations, {} free bikes", stations_.size(),
      free_bikes_.size());
  }

  msg_ptr make_ppr_request(geo::latlng const& one,
                           std::vector<geo::latlng> const& many,
                           SearchDir const dir, double const max_duration) {
    auto const fbs_pos = to_fbs(one);
    message_creator mc;
    mc.create_and_finish(MsgContent_FootRoutingRequest,
                         CreateFootRoutingRequest(
                             mc, &fbs_pos,
                             mc.CreateVectorOfStructs(utl::to_vec(
                                 many, [](auto&& p) { return to_fbs(p); })),
                             ppr::CreateSearchOptions(
                                 mc, mc.CreateString("default"), max_duration),
                             dir, false, false, false)
                             .Union(),
                         "/ppr/route");
    return make_msg(mc);
  }

  msg_ptr make_osrm_request(geo::latlng const& one,
                            std::vector<geo::latlng> const& many,
                            SearchDir direction) {
    auto const fbs_pos = to_fbs(one);
    message_creator mc;
    mc.create_and_finish(MsgContent_OSRMOneToManyRequest,
                         osrm::CreateOSRMOneToManyRequest(
                             mc, mc.CreateString("bike"), direction, &fbs_pos,
                             mc.CreateVectorOfStructs(utl::to_vec(
                                 many, [](auto&& p) { return to_fbs(p); })))
                             .Union(),
                         "/osrm/one_to_many");
    return make_msg(mc);
  }

  msg_ptr route(schedule const&, msg_ptr const& m) {
    using osrm::OSRMOneToManyResponse;
    using ppr::FootRoutingResponse;

    struct journey {
      struct invalid {};
      struct s {  // station bound
        uint32_t sx_, sp_, p_;
      };
      struct b {  // free-float
        uint32_t b_, p_;
      };

      bool valid() const { return !std::holds_alternative<invalid>(info_); }

      uint16_t duration_{std::numeric_limits<uint16_t>::max()};
      uint16_t distance_{std::numeric_limits<uint16_t>::max()};
      std::variant<invalid, s, b> info_{invalid{}};
    };

    constexpr auto const max_walk_speed = 1.1;  // m/s 4km/h
    constexpr auto const max_bike_speed = 7.0;  // m/s 25km/h

    auto const req = motis_content(GBFSRoutingRequest, m);

    auto const max_walk_duration = req->max_foot_duration();
    auto const max_bike_duration = req->max_bike_duration();
    auto const max_walk_dist = max_walk_duration * 60 * max_walk_speed;
    auto const max_bike_dist = req->max_bike_duration() * 60 * max_bike_speed;
    auto const max_total_dist = max_walk_dist + max_bike_dist;

    auto const x = from_fbs(req->x());

    auto const p = pt_stations_rtree_.in_radius(x, max_total_dist);
    auto const p_pos = utl::to_vec(p, [&](auto const idx) {
      auto const& s = *sched_.stations_.at(idx);
      return geo::latlng{s.lat(), s.lng()};
    });

    auto const sx = stations_rtree_.in_radius(x, max_walk_dist);
    auto const sx_pos =
        utl::to_vec(sx, [&](auto const idx) { return stations_.at(idx).pos_; });
    auto const sp = std::accumulate(
        begin(p_pos), end(p_pos), std::vector<size_t>{},
        [&](std::vector<size_t> acc, geo::latlng const& pt_station_pos) {
          return utl::concat(
              acc, stations_rtree_.in_radius(pt_station_pos, max_walk_dist));
        });
    auto const sp_pos =
        utl::to_vec(sp, [&](auto const idx) { return stations_.at(idx).pos_; });
    auto const b = [&]() {
      if (req->dir() == SearchDir_Forward) {
        return free_bikes_rtree_.in_radius(x, max_walk_dist);
      } else {
        return std::accumulate(
            begin(p), end(p), std::vector<size_t>{},
            [&](std::vector<size_t> acc, size_t const idx) {
              auto const* s = sched_.stations_.at(idx).get();
              return utl::concat(acc, stations_rtree_.in_radius(
                                          {s->lat(), s->lng()}, max_walk_dist));
            });
      }
    }();
    auto const b_pos = utl::to_vec(
        b, [&](auto const idx) { return free_bikes_.at(idx).pos_; });

    auto p_best_journeys = std::vector<journey>{};
    p_best_journeys.reserve(p.size());
    if (req->dir() == SearchDir_Forward) {
      // free-float FWD: x --walk--> [b] --bike--> [p]
      auto const f_x_to_b_walks = motis_call(make_ppr_request(
          x, b_pos, SearchDir_Forward, req->max_foot_duration()));
      auto const f_b_rides = utl::to_vec(b, [&](auto const& free_bike_idx) {
        return motis_call(make_osrm_request(free_bikes_.at(free_bike_idx).pos_,
                                            p_pos, SearchDir_Forward));
      });

      // station FWD: x --walk--> [sx] --bike--> [sp] --walk--> [p]
      auto const f_x_to_sx_walks = motis_call(make_ppr_request(
          x, sx_pos, SearchDir_Forward, req->max_foot_duration()));
      auto const f_sx_to_sp_rides = utl::to_vec(sx, [&](auto const& sx_index) {
        return motis_call(make_osrm_request(stations_.at(sx_index).pos_, sp_pos,
                                            SearchDir_Forward));
      });
      auto const f_sp_to_p_walks = utl::to_vec(sp_pos, [&](auto const& pos) {
        return motis_call(make_ppr_request(pos, p_pos, SearchDir_Forward,
                                           req->max_foot_duration()));
      });

      for (auto const [sx_idx, x_to_sx_res] : utl::enumerate(
               *motis_content(FootRoutingResponse, f_x_to_sx_walks->val())
                    ->routes())) {
        if (x_to_sx_res->routes()->size() == 0) {
          continue;
        }
        auto const x_to_sx_route = x_to_sx_res->routes()->Get(0);
        auto const x_to_sx_distance = x_to_sx_route->distance();
        auto const x_to_sx_walk_duration = x_to_sx_route->duration();
        if (x_to_sx_walk_duration > max_walk_duration) {
          continue;
        }

        for (auto const& [sp_idx, sx_to_sp_res] :
             utl::enumerate(*motis_content(OSRMOneToManyResponse,
                                           f_sx_to_sp_rides.at(sx_idx)->val())
                                 ->costs())) {
          auto const sx_to_sp_duration = sx_to_sp_res->duration();
          auto const sx_to_sp_distance = sx_to_sp_res->distance();
          if (sx_to_sp_duration > max_bike_duration) {
            continue;
          }

          for (auto const& [p_idx, sp_to_p_res] :
               utl::enumerate(*motis_content(FootRoutingResponse,
                                             f_sp_to_p_walks.at(sp_idx)->val())
                                   ->routes())) {
            if (sp_to_p_res->routes()->size() == 0) {
              continue;
            }
            auto const sp_to_p_route = x_to_sx_res->routes()->Get(0);
            auto const sp_to_p_distance = sp_to_p_route->distance();
            auto const sp_to_p_walk_duration = sp_to_p_route->duration();
            if (sp_to_p_walk_duration > max_walk_duration ||
                x_to_sx_walk_duration + sp_to_p_walk_duration >
                    max_walk_duration) {
              continue;
            }

            auto const total_duration = x_to_sx_walk_duration +
                                        sx_to_sp_duration +
                                        sp_to_p_walk_duration;
            if (auto& best = p_best_journeys[p_idx];
                best.duration_ > total_duration) {
              best.duration_ = total_duration;
              best.distance_ =
                  x_to_sx_distance + sx_to_sp_distance + sp_to_p_distance;
              best.info_ = journey::s{static_cast<uint32_t>(sx_idx),
                                      static_cast<uint32_t>(sp_idx),
                                      static_cast<uint32_t>(p_idx)};
            }
          }
        }
      }
    } else {
      // TODO(felix)
      // BWD
      //   free-float BWD: [p] --walk--> [b] --bike--> x
      //   station BWD: [p] --walk--> [sp] --bike--> [sx] --walk--> x
    }

    return {};
  }

  config const& config_;
  schedule const& sched_;
  std::mutex mutex_;
  std::vector<station> stations_;
  std::vector<free_bike> free_bikes_;
  geo::point_rtree stations_rtree_, free_bikes_rtree_, pt_stations_rtree_;
};

gbfs::gbfs() : module("RIS", "gbfs") {
  param(config_.update_interval_minutes_, "update_interval",
        "update interval in minutes");
  param(config_.urls_, "urls", "URLs to fetch data from");
}

gbfs::~gbfs() = default;

void gbfs::import(import_dispatcher& reg) {
  std::make_shared<event_collector>(
      get_data_directory().generic_string(), "parking", reg,
      [this](event_collector::dependencies_map_t const&,
             event_collector::publish_fn_t const&) {
        import_successful_ = true;
      })
      ->require("SCHEDULE",
                [](msg_ptr const& msg) {
                  return msg->get()->content_type() == MsgContent_ScheduleEvent;
                })
      ->require("OSRM", [](msg_ptr const& msg) {
        using import::OSRMEvent;
        return msg->get()->content_type() == MsgContent_OSRMEvent &&
               motis_content(OSRMEvent, msg)->profile()->str() == "bike";
      });
}

void gbfs::init(motis::module::registry& r) {
  impl_ = std::make_unique<impl>(config_, get_sched());
  r.subscribe("/init", [&]() { impl_->init(get_sched()); });
  r.register_op("/gbfs/route",
                [&](msg_ptr const& m) { return impl_->route(get_sched(), m); });
}

}  // namespace motis::gbfs
