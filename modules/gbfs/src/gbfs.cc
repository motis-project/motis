#include "motis/gbfs/gbfs.h"

#include <mutex>
#include <numeric>

#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

#include "utl/concat.h"
#include "utl/enumerate.h"
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

  static msg_ptr make_one_to_many(char const* profile, geo::latlng const& one,
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

  static msg_ptr make_table_request(char const* profile,
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

  msg_ptr route(schedule const&, msg_ptr const& m) {
    using osrm::OSRMManyToManyResponse;
    using osrm::OSRMOneToManyResponse;

    struct journey {
      struct invalid {};
      struct s {  // station bound
        uint32_t sx_, sp_, p_;
      };
      struct b {  // free-float
        uint32_t b_, p_;
      };

      bool valid() const { return !std::holds_alternative<invalid>(info_); }

      uint16_t total_duration_{std::numeric_limits<uint16_t>::max()};
      uint16_t walk_duration_{std::numeric_limits<uint16_t>::max()};
      uint16_t bike_duration_{std::numeric_limits<uint16_t>::max()};
      double bike_distance_{std::numeric_limits<double>::max()};
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
              auto const& s = *sched_.stations_.at(idx);
              acc.emplace_back(free_bikes_rtree_.nearest({s.lat(), s.lng()}, 1U)
                                   .at(0)
                                   .second);
              return acc;
            });
      }
    }();
    auto const b_pos = utl::to_vec(
        b, [&](auto const idx) { return free_bikes_.at(idx).pos_; });

    auto p_best_journeys = std::vector<journey>{};
    p_best_journeys.resize(p.size());

    if (req->dir() == SearchDir_Forward) {
      // REQUESTS
      // free-float FWD: x --walk--> [b] --bike--> [p]
      auto const f_x_to_b_walks =
          motis_call(make_one_to_many("foot", x, b_pos, SearchDir_Forward));
      auto const f_b_to_p_rides =
          motis_call(make_table_request("bike", b_pos, p_pos));

      // REQUESTS
      // station BWD: [p] --walk--> [sp] --bike--> [sx] --walk--> x
      auto const f_x_to_sx_walks =
          motis_call(make_one_to_many("foot", x, sx_pos, SearchDir_Forward));
      auto const f_sx_to_sp_rides =
          motis_call(make_table_request("bike", sx_pos, sp_pos));
      auto const f_sp_to_p_walks =
          motis_call(make_table_request("foot", sp_pos, p_pos));

      // BUILD JOURNEYS
      // free-float FWD: x --walk--> [b] --bike--> [p]
      auto const b_to_p_table =
          motis_content(OSRMManyToManyResponse, f_b_to_p_rides->val())->costs();
      for (auto const& [b_vec_idx, x_to_b_res] : utl::enumerate(
               *motis_content(OSRMOneToManyResponse, f_x_to_b_walks->val())
                    ->costs())) {
        auto const x_to_b_walk_duration = x_to_b_res->duration() / 60.0;
        if (x_to_b_walk_duration > max_walk_duration) {
          continue;
        }

        for (auto const& [p_vec_idx, p_id] : utl::enumerate(p)) {
          auto const b_to_p_duration =
              b_to_p_table->Get(b_vec_idx * p.size() + p_vec_idx);
          if (b_to_p_duration > max_bike_duration) {
            continue;
          }

          auto const total_duration = x_to_b_walk_duration + b_to_p_duration;
          if (auto& best = p_best_journeys[p_vec_idx];
              best.total_duration_ > total_duration) {
            best.total_duration_ = total_duration;
            best.walk_duration_ = x_to_b_walk_duration;
            best.bike_duration_ = b_to_p_duration;
            best.info_ = journey::b{static_cast<uint32_t>(b_vec_idx),
                                    static_cast<uint32_t>(p_vec_idx)};
          }
        }
      }

      // BUILD JOURNEYS
      // station BWD: [p] --walk--> [sp] --bike--> [sx] --walk--> x
      auto const sx_to_sp_table =
          motis_content(OSRMManyToManyResponse, f_sx_to_sp_rides->val())
              ->costs();
      auto const sp_to_p_table =
          motis_content(OSRMManyToManyResponse, f_sp_to_p_walks->val())
              ->costs();
      for (auto const [sx_vec_idx, x_to_sx_res] : utl::enumerate(
               *motis_content(OSRMOneToManyResponse, f_x_to_sx_walks->val())
                    ->costs())) {
        auto const x_to_sx_walk_duration = x_to_sx_res->duration() / 60.0;
        if (x_to_sx_walk_duration > max_walk_duration) {
          continue;
        }

        for (auto const& [sp_vec_idx, sp_id] : utl::enumerate(sp)) {
          auto const sx_to_sp_duration =
              sx_to_sp_table->Get(sx_vec_idx * sp.size() + sp_vec_idx) / 60.0;
          if (sx_to_sp_duration > max_bike_duration) {
            continue;
          }

          for (auto const& [p_vec_idx, p_id] : utl::enumerate(p)) {
            auto const sp_to_p_walk_duration =
                sp_to_p_table->Get(sp_vec_idx * p.size() + p_vec_idx) / 60.0;
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
              best.walk_duration_ =
                  x_to_sx_walk_duration + sp_to_p_walk_duration;
              best.bike_duration_ = sx_to_sp_duration;
              best.info_ = journey::s{static_cast<uint32_t>(sx_vec_idx),
                                      static_cast<uint32_t>(sp_vec_idx),
                                      static_cast<uint32_t>(p_vec_idx)};
            }
          }
        }
      }
    } else {
      // REQUESTS
      // free-float BWD: [p] --walk--> [b] --bike--> x
      auto const f_p_to_b_walks =
          motis_call(make_table_request("foot", p_pos, b_pos));
      auto const f_b_to_x_rides =
          motis_call(make_one_to_many("bike", x, b_pos, SearchDir_Backward));

      // REQUESTS
      // station BWD: [p] --walk--> [sp] --bike--> [sx] --walk--> x
      auto const f_p_to_sp_walks =
          motis_call(make_table_request("foot", p_pos, sp_pos));
      auto const f_sp_to_sx_rides =
          motis_call(make_table_request("bike", sp_pos, sx_pos));
      auto const f_sx_to_x_walks =
          motis_call(make_one_to_many("foot", x, sx_pos, SearchDir_Backward));

      // BUILD JOURNEYS
      // free-float BWD: [p] --walk--> [b] --bike--> x
      auto const p_to_b_table =
          motis_content(OSRMManyToManyResponse, f_p_to_b_walks->val())->costs();
      for (auto const& [b_vec_idx, b_to_x_res] : utl::enumerate(
               *motis_content(OSRMOneToManyResponse, f_b_to_x_rides->val())
                    ->costs())) {
        auto const b_to_x_bike_duration = b_to_x_res->duration() / 60.0;
        if (b_to_x_bike_duration > max_bike_duration) {
          continue;
        }

        for (auto const& [p_vec_idx, p_id] : utl::enumerate(p)) {
          auto const p_to_b_walk_duration =
              p_to_b_table->Get(b_vec_idx * p.size() + p_vec_idx);
          if (p_to_b_walk_duration > max_walk_duration) {
            continue;
          }

          auto const total_duration =
              p_to_b_walk_duration + b_to_x_bike_duration;
          if (auto& best = p_best_journeys[p_vec_idx];
              best.total_duration_ > total_duration) {
            best.total_duration_ = total_duration;
            best.walk_duration_ = p_to_b_walk_duration;
            best.bike_duration_ = b_to_x_bike_duration;
            best.info_ = journey::b{static_cast<uint32_t>(b_vec_idx),
                                    static_cast<uint32_t>(p_vec_idx)};
          }
        }
      }

      // BUILD JOURNEYS
      // station BWD: [p] --walk--> [sp] --bike--> [sx] --walk--> x
      auto const p_to_sp_table =
          motis_content(OSRMManyToManyResponse, f_p_to_sp_walks->val())
              ->costs();
      auto const sp_to_sx_table =
          motis_content(OSRMManyToManyResponse, f_sp_to_sx_rides->val())
              ->costs();
      for (auto const [sx_vec_idx, sx_to_x_res] : utl::enumerate(
               *motis_content(OSRMOneToManyResponse, f_sx_to_x_walks->val())
                    ->costs())) {
        auto const sx_to_x_walk_duration = sx_to_x_res->duration() / 60.0;
        if (sx_to_x_walk_duration > max_walk_duration) {
          continue;
        }

        for (auto const& [sp_vec_idx, sp_id] : utl::enumerate(sp)) {
          auto const sp_to_sx_duration =
              sp_to_sx_table->Get(sx_vec_idx * sp.size() + sp_vec_idx) / 60.0;
          if (sp_to_sx_duration > max_bike_duration) {
            continue;
          }

          for (auto const& [p_vec_idx, p_id] : utl::enumerate(p)) {
            auto const p_to_sx_walk_duration =
                p_to_sp_table->Get(p_vec_idx * sp.size() + sp_vec_idx) / 60.0;
            if (p_to_sx_walk_duration > max_walk_duration ||
                p_to_sx_walk_duration + sx_to_x_walk_duration >
                    max_walk_duration) {
              continue;
            }

            auto const total_duration = p_to_sx_walk_duration +
                                        sp_to_sx_duration +
                                        sx_to_x_walk_duration;
            if (auto& best = p_best_journeys[p_vec_idx];
                best.total_duration_ > total_duration) {
              best.total_duration_ = total_duration;
              best.walk_duration_ =
                  p_to_sx_walk_duration + sx_to_x_walk_duration;
              best.bike_duration_ = sp_to_sx_duration;
              best.info_ = journey::s{static_cast<uint32_t>(sx_vec_idx),
                                      static_cast<uint32_t>(sp_vec_idx),
                                      static_cast<uint32_t>(p_vec_idx)};
            }
          }
        }
      }
    }

    message_creator fbb;
    auto const routes =
        utl::all(p_best_journeys)  //
        | utl::remove_if([](journey const& j) { return !j.valid(); })  //
        |
        utl::transform([&](journey const& j) {
          return std::visit(
              utl::overloaded{
                  [&](journey::invalid const&) {
                    throw std::runtime_error{"unreachable"};
                    return fbs::Offset<RouteInfo>{};
                  },

                  [&](journey::b const& free_bike_info) {
                    auto const& free_bike =
                        free_bikes_.at(b.at(free_bike_info.b_));
                    auto const& station =
                        *sched_.stations_.at(p.at(free_bike_info.p_));
                    auto const pos = to_fbs(free_bike.pos_);
                    return CreateRouteInfo(
                        fbb, to_fbs(fbb, station), BikeRoute_FreeBikeRoute,
                        CreateFreeBikeRoute(
                            fbb, fbb.CreateString(free_bike.id_), &pos)
                            .Union(),
                        j.total_duration_, j.walk_duration_, j.bike_duration_);
                  },

                  [&](journey::s const& station_bike_info) {
                    auto const& sx_bike_station =
                        stations_.at(sx.at(station_bike_info.sx_));
                    auto const& sp_bike_station =
                        stations_.at(sp.at(station_bike_info.sp_));
                    auto const& station =
                        *sched_.stations_.at(p.at(station_bike_info.p_));
                    return CreateRouteInfo(
                        fbb, to_fbs(fbb, station), BikeRoute_StationBikeRoute,
                        CreateStationBikeRoute(
                            fbb,
                            CreateGBFSStation(
                                fbb, fbb.CreateString(sx_bike_station.id_)),
                            CreateGBFSStation(
                                fbb, fbb.CreateString(sp_bike_station.id_)))
                            .Union(),
                        j.total_duration_, j.walk_duration_, j.bike_duration_);
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

  msg_ptr geo_json(msg_ptr const&) {
    auto sb = rapidjson::StringBuffer{};
    auto w = rapidjson::Writer<rapidjson::StringBuffer>{sb};

    w.StartObject();

    w.String("type");
    w.String("FeatureCollection");

    w.String("features");
    {
      w.StartArray();

      {
        w.StartObject();

        w.String("type");
        w.String("Feature");

        w.String("geometry");
        {
          w.StartObject();

          w.String("type");
          w.String("MultiPoint");

          w.String("coordinates");
          {
            w.StartArray();

            for (auto const& s : stations_) {
              w.StartArray();
              w.Double(s.pos_.lng_);
              w.Double(s.pos_.lat_);
              w.EndArray();
            }

            w.EndArray();
          }

          w.EndObject();
        }

        w.String("properties");
        {
          w.StartObject();

          w.String("marker-color");
          w.String("red");

          w.String("name");
          w.String("");

          w.EndObject();
        }

        w.EndObject();
      }

      {
        w.StartObject();

        w.String("type");
        w.String("Feature");

        w.String("geometry");
        {
          w.StartObject();

          w.String("type");
          w.String("MultiPoint");

          w.String("coordinates");
          {
            w.StartArray();

            for (auto const& [i, s] : utl::enumerate(free_bikes_)) {
              if (i % 3 == 0) {
                w.StartArray();
                w.Double(s.pos_.lng_);
                w.Double(s.pos_.lat_);
                w.EndArray();
              }
            }

            w.EndArray();
          }

          w.EndObject();
        }

        w.String("properties");
        {
          w.StartObject();

          w.String("marker-color");
          w.String("blue");

          w.EndObject();
        }

        w.EndObject();
      }

      w.EndArray();
    }

    w.EndObject();

    message_creator fbb;
    fbb.create_and_finish(
        MsgContent_GBFSGeoJSONResponse,
        CreateGBFSGeoJSONResponse(
            fbb, fbb.CreateString(sb.GetString(), sb.GetLength()))
            .Union());

    return make_msg(fbb);
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
  r.register_op("/gbfs/geojson",
                [&](msg_ptr const& m) { return impl_->geo_json(m); });
}

}  // namespace motis::gbfs
