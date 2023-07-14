#include "motis/gbfs/gbfs.h"

#include <filesystem>
#include <mutex>
#include <numeric>

#include "utl/concat.h"
#include "utl/enumerate.h"
#include "utl/erase_duplicates.h"
#include "utl/get_or_create.h"
#include "utl/pipes.h"

#include "geo/point_rtree.h"

#include "tiles/db/clear_database.h"
#include "tiles/db/feature_inserter_mt.h"
#include "tiles/feature/feature.h"
#include "tiles/fixed/convert.h"
#include "tiles/get_tile.h"
#include "tiles/parse_tile_url.h"

#include "motis/core/common/logging.h"
#include "motis/core/schedule/station_lookup.h"
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
namespace fs = std::filesystem;
using namespace motis::logging;
using namespace motis::module;

namespace motis::gbfs {

constexpr auto const bike_ready_time = 3;

struct journey {
  struct invalid {};
  struct s {  // station bound
    friend std::ostream& operator<<(std::ostream& out, s const& x) {
      return out << "(STATION_BIKE: first_walk_duration="
                 << x.first_walk_duration_
                 << ", bike_duration=" << x.bike_duration_
                 << ", second_walk_duration=" << x.second_walk_duration_ << ")";
    }
    duration first_walk_duration_{0};
    duration bike_duration_{0};
    duration second_walk_duration_{0};
    uint32_t sx_, sp_, p_;
  };
  struct b {  // free-float
    friend std::ostream& operator<<(std::ostream& out, b const& x) {
      return out << "(FREE_FLOAT: first_walk_duration=" << x.walk_duration_
                 << ", bike_duration=" << x.bike_duration_ << ")";
    }
    duration walk_duration_{0};
    duration bike_duration_{0};
    uint32_t b_, p_;
  };

  bool valid() const { return !std::holds_alternative<invalid>(info_); }

  uint16_t total_duration_{std::numeric_limits<uint16_t>::max()};
  std::variant<invalid, s, b> info_{invalid{}};
};

struct gbfs::impl {
  explicit impl(fs::path data_dir, config const& c, station_lookup const& st)
      : config_{c}, st_{st}, data_dir_{std::move(data_dir)} {}

  void fetch_stream(std::string url) {
    auto tag = std::string{"default"};
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
    auto& info = utl::get_or_create(status_, tag, [&]() {
      return provider_status{std::make_unique<tiles_database>(
          (data_dir_ / (tag + "tiles.mdb")).string(), config_.db_size_)};
    });
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

    info.tiles_->clear();

    tiles::layer_names_builder layer_names;
    auto const free_bike_layer_id = layer_names.get_layer_idx("vehicle");
    auto const station_bike_layer_id = layer_names.get_layer_idx("station");

    static constexpr auto const kMinZoomLevel = 10;
    auto feature_inserter = tiles::feature_inserter_mt{
        tiles::dbi_handle{info.tiles_->db_handle_,
                          info.tiles_->db_handle_.features_dbi_opener()},
        info.tiles_->pack_handle_};

    for (auto const& [idx, nfo] : utl::enumerate(info.free_bikes_)) {
      tiles::feature f;
      f.id_ = idx;
      f.layer_ = free_bike_layer_id;
      f.zoom_levels_ = {kMinZoomLevel, tiles::kMaxZoomLevel};
      f.meta_.emplace_back("type", tiles::encode_string(vehicle_type));
      f.meta_.emplace_back("tag", tiles::encode_string(tag));
      f.meta_.emplace_back("id", tiles::encode_string(nfo.id_));
      f.geometry_ = tiles::fixed_point{
          {tiles::latlng_to_fixed({nfo.pos_.lat_, nfo.pos_.lng_})}};
      feature_inserter.insert(f);
    }

    for (auto const& [idx, nfo] : utl::enumerate(info.stations_)) {
      tiles::feature f;
      f.id_ = idx;
      f.layer_ = station_bike_layer_id;
      f.zoom_levels_ = {kMinZoomLevel, tiles::kMaxZoomLevel};
      f.meta_.emplace_back("type", tiles::encode_string(vehicle_type));
      f.meta_.emplace_back("tag", tiles::encode_string(tag));
      f.meta_.emplace_back("name", tiles::encode_string(nfo.name_));
      f.meta_.emplace_back("id", tiles::encode_string(nfo.id_));
      f.meta_.emplace_back("vehicles_available",
                           tiles::encode_integer(nfo.bikes_available_));
      f.geometry_ = tiles::fixed_point{
          {tiles::latlng_to_fixed({nfo.pos_.lat_, nfo.pos_.lng_})}};
      feature_inserter.insert(f);
    }

    {
      auto txn = info.tiles_->db_handle_.make_txn();
      layer_names.store(info.tiles_->db_handle_, txn);
      txn.commit();
    }

    info.tiles_->render_ctx_ = tiles::make_render_ctx(info.tiles_->db_handle_);
  }

  void init() {
    auto const t = scoped_timer{"GBFS init"};
    if (!config_.urls_.empty()) {
      fs::create_directories(data_dir_);
    }

    motis_parallel_for(config_.urls_, [&](auto&& url) { fetch_stream(url); });

    auto const lock = std::scoped_lock{mutex_};
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

  msg_ptr route(msg_ptr const& m) {
    using osrm::OSRMManyToManyResponse;
    using osrm::OSRMOneToManyResponse;

    constexpr auto const max_walk_speed = 1.1;  // m/s 4km/h
    constexpr auto const max_bike_speed = 7.0;  // m/s 25km/h
    constexpr auto const max_car_speed = 27.8;  // m/s 100km/h

    auto const req = motis_content(GBFSRoutingRequest, m);

    auto const provider = req->provider()->str();
    auto const status_it = status_.find(provider);
    utl::verify(status_it != end(status_), "provider {} not found", provider);

    auto const lock = std::lock_guard{mutex_};
    auto const& info = status_it->second;
    auto const& stations = info.stations_;
    auto const& stations_rtree = info.stations_rtree_;
    auto const& free_bikes = info.free_bikes_;
    auto const& free_bikes_rtree = info.free_bikes_rtree_;
    auto const& vehicle_type = info.vehicle_type_;
    utl::verify(vehicle_type == "car" || vehicle_type == "bike",
                "unsupported vehicle type {}", vehicle_type);

    auto const max_walk_duration = req->max_foot_duration();
    auto const max_bike_duration = req->max_bike_duration();
    auto const max_walk_dist = max_walk_duration * 60 * max_walk_speed;
    auto const max_bike_dist =
        req->max_bike_duration() * 60 *
        (vehicle_type == "bike" ? max_bike_speed : max_car_speed);
    auto const max_total_dist = max_walk_dist + max_bike_dist;

    auto const x = from_fbs(req->x());

    auto const p = st_.in_radius(x, max_total_dist);
    auto p_pos =
        utl::to_vec(p, [&](std::pair<lookup_station, double> const& el) {
          return el.first.pos_;
        });
    utl::concat(p_pos, utl::to_vec(*req->direct(), [](Position const* p) {
                  return from_fbs(p);
                }));
    if (p_pos.empty() && req->direct()->size() == 0U) {
      l(logging::debug, "no stations found in {}km radius around {}",
        max_total_dist / 1000.0, x);
      return empty_response(req->dir());
    }

    auto const sx = stations_rtree.in_radius(x, max_walk_dist);
    auto sp = std::accumulate(
        begin(p_pos), end(p_pos), std::vector<size_t>{},
        [&](std::vector<size_t> acc, geo::latlng const& pt_station_pos) {
          return utl::concat(
              acc, stations_rtree.in_radius(pt_station_pos, max_walk_dist));
        });
    auto b = [&]() {
      if (req->dir() == SearchDir_Forward) {
        return free_bikes_rtree.in_radius(x, max_walk_dist);
      } else {
        return std::accumulate(
            begin(p_pos), end(p_pos), std::vector<size_t>{},
            [&](std::vector<size_t> acc, geo::latlng const& pos) {
              auto const closest = free_bikes_rtree.nearest(pos, 1U);
              if (!closest.empty()) {
                acc.emplace_back(closest.at(0).second);
              }
              return acc;
            });
      }
    }();

    if (b.empty() && (sp.empty() || sx.empty())) {
      l(logging::debug,
        "no free bikes found, no stations found (max_bike_dist={}), "
        "(max_walk_dist={})",
        max_walk_dist, max_bike_dist);
      return empty_response(req->dir());
    }

    auto const sx_pos =
        utl::to_vec(sx, [&](auto const idx) { return stations.at(idx).pos_; });

    utl::erase_duplicates(b);
    auto const b_pos =
        utl::to_vec(b, [&](auto const idx) { return free_bikes.at(idx).pos_; });

    utl::erase_duplicates(sp);
    auto const sp_pos =
        utl::to_vec(sp, [&](auto const idx) { return stations.at(idx).pos_; });

    auto p_best_journeys = std::vector<journey>{};
    p_best_journeys.resize(p_pos.size());

    auto d_best_journeys = std::vector<journey>{};
    d_best_journeys.resize(req->direct()->size());

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
      // station FWD: x --walk--> [sx] --bike--> [sp] --walk--> [p]
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
        auto const x_to_b_costs =
            motis_content(OSRMOneToManyResponse, f_x_to_b_walks->val())
                ->costs();

        for (auto const& [b_vec_idx, x_to_b_res] :
             utl::enumerate(*x_to_b_costs)) {
          auto const x_to_b_walk_duration =
              static_cast<duration>(std::ceil(x_to_b_res->duration() / 60.0));
          if (x_to_b_walk_duration > max_walk_duration) {
            continue;
          }

          for (auto const& [p_vec_idx, _] : utl::enumerate(p_pos)) {
            auto const b_to_p_duration = static_cast<duration>(std::ceil(
                b_to_p_table->Get(b_vec_idx * p_pos.size() + p_vec_idx) /
                60.0));
            if (b_to_p_duration > max_bike_duration) {
              continue;
            }

            auto const total_duration =
                bike_ready_time + x_to_b_walk_duration + b_to_p_duration;
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
              static_cast<duration>(std::ceil(x_to_sx_res->duration() / 60.0));
          if (x_to_sx_walk_duration > max_walk_duration) {
            continue;
          }

          for (auto const& [sp_vec_idx, sp_id] : utl::enumerate(sp)) {
            auto const sx_to_sp_duration = static_cast<duration>(std::ceil(
                sx_to_sp_table->Get(sx_vec_idx * sp.size() + sp_vec_idx) /
                60.0));
            if (sx_to_sp_duration > max_bike_duration) {
              continue;
            }

            for (auto const& [p_vec_idx, _] : utl::enumerate(p_pos)) {
              auto const sp_to_p_walk_duration = static_cast<duration>(
                  std::ceil(sp_to_p_table->Get(sp_vec_idx * p_pos.size() +
                                               p_vec_idx) /
                            60.0));
              if (sp_to_p_walk_duration > max_walk_duration ||
                  x_to_sx_walk_duration + sp_to_p_walk_duration >
                      max_walk_duration) {
                continue;
              }

              auto const total_duration =
                  bike_ready_time + x_to_sx_walk_duration + sx_to_sp_duration +
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
              static_cast<duration>(std::ceil(b_to_x_res->duration() / 60.0));
          if (b_to_x_bike_duration > max_bike_duration) {
            continue;
          }

          for (auto const& [p_vec_idx, p_id] : utl::enumerate(p)) {
            auto const p_to_b_walk_duration = static_cast<duration>(std::ceil(
                p_to_b_table->Get(p_vec_idx * b.size() + b_vec_idx) / 60.0));
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
              static_cast<duration>(std::ceil(sx_to_x_res->duration() / 60.0));
          if (sx_to_x_walk_duration > max_walk_duration) {
            continue;
          }

          for (auto const& [sp_vec_idx, sp_id] : utl::enumerate(sp)) {
            if (stations.at(sp.at(sp_vec_idx)).bikes_available_ == 0) {
              continue;
            }

            auto const sp_to_sx_duration = static_cast<duration>(std::ceil(
                sp_to_sx_table->Get(sp_vec_idx * sx.size() + sx_vec_idx) /
                60.0));

            if (sp_to_sx_duration > max_bike_duration) {
              continue;
            }

            for (auto const& [p_vec_idx, p_id] : utl::enumerate(p)) {
              auto const p_to_sp_walk_duration =
                  static_cast<duration>(std::ceil(
                      p_to_sp_table->Get(p_vec_idx * sp.size() + sp_vec_idx) /
                      60.0));
              if (p_to_sp_walk_duration > max_walk_duration ||
                  p_to_sp_walk_duration + sx_to_x_walk_duration >
                      max_walk_duration) {
                continue;
              }

              auto const total_duration =
                  bike_ready_time + p_to_sp_walk_duration + sp_to_sx_duration +
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
        |
        utl::transform([&](journey const& j) {
          return std::visit(
              utl::overloaded{
                  [&](journey::invalid const&) -> fbs::Offset<RouteInfo> {
                    throw std::runtime_error{"unreachable"};
                  },

                  [&](journey::b const& free_bike_info) {
                    auto const& free_bike =
                        free_bikes.at(b.at(free_bike_info.b_));
                    auto const pos = to_fbs(free_bike.pos_);
                    return CreateRouteInfo(
                        fbb, fbb.CreateString(vehicle_type),
                        free_bike_info.p_ < p.size() ? P_Station : P_Direct,
                        free_bike_info.p_ < p.size()
                            ? p.at(free_bike_info.p_).first.to_fbs(fbb).Union()
                            : CreateDirect(
                                  fbb, req->direct()->Get(p.size() -
                                                          free_bike_info.p_))
                                  .Union(),
                        BikeRoute_FreeBikeRoute,
                        CreateFreeBikeRoute(
                            fbb, fbb.CreateString(free_bike.id_), &pos,
                            free_bike_info.walk_duration_ + bike_ready_time,
                            free_bike_info.bike_duration_)
                            .Union(),
                        j.total_duration_);
                  },

                  [&](journey::s const& station_bike_info) {
                    auto const& sx_bike_station =
                        stations.at(sx.at(station_bike_info.sx_));
                    auto const& sp_bike_station =
                        stations.at(sp.at(station_bike_info.sp_));
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
                        station_bike_info.p_ < p.size() ? P_Station : P_Direct,
                        station_bike_info.p_ < p.size()
                            ? p.at(station_bike_info.p_)
                                  .first.to_fbs(fbb)
                                  .Union()
                            : CreateDirect(
                                  fbb, req->direct()->Get(p.size() -
                                                          station_bike_info.p_))
                                  .Union(),
                        BikeRoute_StationBikeRoute,
                        CreateStationBikeRoute(
                            fbb,
                            req->dir() == SearchDir_Forward ? sx_gbfs_station
                                                            : sp_gbfs_station,
                            req->dir() == SearchDir_Forward ? sp_gbfs_station
                                                            : sx_gbfs_station,
                            station_bike_info.first_walk_duration_ +
                                bike_ready_time,
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

  msg_ptr tiles(msg_ptr const& msg) {
    auto const split_target = [](std::string_view const target) {
      static constexpr auto const prefix = std::string_view{"/gbfs/tiles/"};
      auto const in = target.substr(prefix.size());
      auto const first_slash = in.find('/');
      utl::verify(first_slash != std::string_view::npos,
                  "invalid gbfs tiles target {}", target);
      return std::pair{std::string{in.substr(0, first_slash)},
                       std::string{in.substr(first_slash)}};
    };

    auto const target = msg->get()->destination()->target();
    auto const [tag, tile_url] =
        split_target({target->c_str(), target->Length()});
    auto const tile = tiles::parse_tile_url(tile_url);
    utl::verify(tile.has_value(), "invalid tile url {}", tile_url);

    tiles::null_perf_counter pc;
    auto& data = status_.at(tag);
    auto const rendered_tile =
        tiles::get_tile(data.tiles_->db_handle_, data.tiles_->pack_handle_,
                        data.tiles_->render_ctx_, *tile, pc);

    message_creator mc;
    std::vector<fbs::Offset<HTTPHeader>> headers;
    fbs::Offset<fbs::String> payload;
    if (rendered_tile) {
      headers.emplace_back(CreateHTTPHeader(
          mc, mc.CreateString("Content-Type"),
          mc.CreateString("application/vnd.mapbox-vector-tile")));
      headers.emplace_back(CreateHTTPHeader(
          mc, mc.CreateString("Content-Encoding"), mc.CreateString("deflate")));
      payload = mc.CreateString(rendered_tile->data(), rendered_tile->size());
    } else {
      payload = mc.CreateString("");
    }

    mc.create_and_finish(
        MsgContent_HTTPResponse,
        CreateHTTPResponse(mc, HTTPStatus_OK, mc.CreateVector(headers), payload)
            .Union());

    return make_msg(mc);
  }

  struct tiles_database {
    explicit tiles_database(std::string const& path, size_t const db_size)
        : db_env_{tiles::make_tile_database(path.c_str(), db_size)},
          db_handle_{db_env_},
          render_ctx_{tiles::make_render_ctx(db_handle_)},
          pack_handle_{path.c_str()} {}

    ~tiles_database() = default;

    tiles_database(tiles_database&&) = delete;
    tiles_database(tiles_database const&) = delete;

    tiles_database& operator=(tiles_database&&) = delete;
    tiles_database& operator=(tiles_database const&) = delete;

    void clear() {
      lmdb::txn txn{db_handle_.env_};
      tiles::clear_database(db_handle_, txn);
      txn.commit();

      pack_handle_.resize(0);
    }

    lmdb::env db_env_;
    tiles::tile_db_handle db_handle_;
    tiles::render_ctx render_ctx_;
    tiles::pack_handle pack_handle_;
  };

  struct provider_status {
    explicit provider_status(std::unique_ptr<tiles_database>&& db)
        : tiles_{std::move(db)} {}
    system_information info_;
    std::string vehicle_type_;
    std::vector<station> stations_;
    std::vector<free_bike> free_bikes_;
    geo::point_rtree free_bikes_rtree_, stations_rtree_;
    std::unique_ptr<tiles_database> tiles_;
  };

  config const& config_;
  station_lookup const& st_;
  std::mutex mutex_;
  std::map<std::string, provider_status> status_;
  fs::path data_dir_;
};

gbfs::gbfs() : module("GBFS", "gbfs") {
  param(config_.update_interval_minutes_, "update_interval",
        "update interval in minutes");
  param(config_.urls_, "urls", "URLs to fetch data from");
  param(config_.db_size_, "db_size", "database size");
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
      ->require("STATIONS",
                [](msg_ptr const& msg) {
                  return msg->get()->content_type() == MsgContent_StationsEvent;
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
  add_shared_data(to_res_id(global_res_id::GBFS_DATA), 0);
  auto const st = get_shared_data<std::shared_ptr<station_lookup>>(
                      to_res_id(global_res_id::STATION_LOOKUP))
                      .get();
  impl_ = std::make_unique<impl>(get_data_directory() / "gbfs", config_, *st);
  r.register_op("/gbfs/route",
                [&](msg_ptr const& m) { return impl_->route(m); },
                {{to_res_id(global_res_id::GBFS_DATA), ctx::access_t::READ}});
  r.register_op("/gbfs/info", [&](msg_ptr const&) { return impl_->info(); },
                {{to_res_id(global_res_id::GBFS_DATA), ctx::access_t::READ}});
  r.register_op("/gbfs/tiles",
                [&](msg_ptr const& m) { return impl_->tiles(m); },
                {{to_res_id(global_res_id::GBFS_DATA), ctx::access_t::READ}});
  r.subscribe(
      "/init",
      [&]() {
        shared_data_->register_timer(
            "GBFS Update",
            boost::posix_time::minutes{config_.update_interval_minutes_},
            [&]() { impl_->init(); },
            {{to_res_id(global_res_id::GBFS_DATA), ctx::access_t::WRITE}});
      },
      {});
}

}  // namespace motis::gbfs
