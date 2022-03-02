#include "motis/gbfs/gbfs.h"

#include <mutex>
#include <numeric>

#include "utl/concat.h"

#include "geo/point_rtree.h"

#include "motis/core/common/logging.h"
#include "motis/core/schedule/schedule.h"
#include "motis/core/conv/position_conv.h"
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
                           ppr::SearchOptions const* search_options,
                           SearchDir dir) {
    assert(search_options != nullptr);
    Position const fbs_position{pos.lat_, pos.lng_};

    message_creator mc;
    mc.create_and_finish(
        MsgContent_FootRoutingRequest,
        CreateFootRoutingRequest(
            mc, &fbs_position,
            mc.CreateVectorOfStructs(utl::to_vec(
                *stations, [](auto&& station) { return *station->pos(); })),
            motis_copy_table(SearchOptions, mc, search_options), dir, false,
            false, false)
            .Union(),
        "/ppr/route");
    return make_msg(mc);
  }

  msg_ptr route(schedule const&, msg_ptr const& m) {
    constexpr auto const max_walk_speed = 1.1;  // m/s 4km/h
    constexpr auto const max_bike_speed = 7.0;  // m/s 25km/h

    auto const req = motis_content(GBFSRoutingRequest, m);

    auto const max_walk_dist = req->max_foot_duration() * 60 * max_walk_speed;
    auto const max_bike_dist = req->max_bike_duration() * 60 * max_bike_speed;
    auto const max_total_dist = max_walk_dist + max_bike_dist;

    auto const x = from_fbs(req->x());
    auto const p = pt_stations_rtree_.in_radius(x, max_total_dist);
    auto const sx = stations_rtree_.in_radius(x, max_walk_dist);
    auto const sp = std::accumulate(
        begin(p), end(p), std::vector<size_t>{},
        [&](std::vector<size_t> acc, size_t const idx) {
          auto const* s = sched_.stations_.at(idx).get();
          return utl::concat(acc, stations_rtree_.in_radius(
                                      {s->lat(), s->lng()}, max_walk_dist));
        });
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

    // FWD
    //   free-float FWD: x --walk--> [b] --bike--> [p]
    //   station FWD: x --walk--> [sx] --bike--> [sp] --walk--> [p]

    // BWD
    //   free-float BWD: [p] --walk--> [b] --bike--> x
    //   station BWD: [p] --walk--> [sp] --bike--> [sx] --walk--> x

    auto const
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
