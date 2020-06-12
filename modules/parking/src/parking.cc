#include "motis/parking/parking.h"

#include <cmath>
#include <limits>
#include <map>
#include <string>

#include "boost/filesystem.hpp"

#include "utl/get_or_create.h"
#include "utl/progress_tracker.h"
#include "utl/to_vec.h"

#include "motis/core/common/logging.h"
#include "motis/core/common/timing.h"
#include "motis/core/access/station_access.h"
#include "motis/core/conv/station_conv.h"
#include "motis/core/statistics/statistics.h"
#include "motis/module/context/get_schedule.h"
#include "motis/module/context/motis_call.h"
#include "motis/module/event_collector.h"
#include "motis/module/ini_io.h"

#include "motis/parking/database.h"
#include "motis/parking/error.h"
#include "motis/parking/mumo_edges.h"
#include "motis/parking/parking_edges.h"
#include "motis/parking/parkings.h"
#include "motis/parking/ppr_profiles.h"
#include "motis/parking/prepare/foot_edges.h"
#include "motis/parking/prepare/parking.h"
#include "motis/parking/prepare/stations.h"
#include "motis/ppr/ppr.h"
#include "motis/ppr/profiles.h"

using namespace motis::module;
using namespace motis::logging;
using namespace motis::osrm;
using namespace motis::ppr;
using namespace flatbuffers;

namespace fs = boost::filesystem;

namespace motis::parking {

struct import_state {
  CISTA_COMPARABLE()
  named<std::string, MOTIS_NAME("osm_path")> osm_path_;
  named<cista::hash_t, MOTIS_NAME("osm_hash")> osm_hash_;
  named<size_t, MOTIS_NAME("osm_size")> osm_size_;
  named<std::string, MOTIS_NAME("ppr_graph_path")> ppr_graph_path_;
  named<cista::hash_t, MOTIS_NAME("ppr_graph_hash")> ppr_graph_hash_;
  named<size_t, MOTIS_NAME("ppr_graph_size")> ppr_graph_size_;
  named<cista::hash_t, MOTIS_NAME("ppr_profiles_hash")> ppr_profiles_hash_;
  named<int, MOTIS_NAME("max_walk_duration")> max_walk_duration_;
  named<cista::hash_t, MOTIS_NAME("schedule_hash")> schedule_hash_;
};

inline geo::latlng to_latlng(Position const* pos) {
  return {pos->lat(), pos->lng()};
}

inline Position to_position(geo::latlng const& loc) {
  return {loc.lat_, loc.lng_};
}

inline Offset<Parking> create_parking(FlatBufferBuilder& fbb,
                                      parking_lot const& p) {
  auto const pos = to_position(p.location_);
  return CreateParking(fbb, p.id_, &pos, p.fee_);
}

inline std::vector<Offset<ParkingEdge>> create_parking_edges(
    FlatBufferBuilder& fbb, std::vector<parking_edges> const& pes) {
  std::map<std::string, Offset<Station>> fbs_stations;
  auto const& sched = get_schedule();
  auto const create_costs = [&](std::vector<parking_edge_costs> const& costs) {
    return fbb.CreateVector(
        utl::to_vec(costs, [&](parking_edge_costs const& c) {
          auto const st =
              utl::get_or_create(fbs_stations, c.station_id_, [&]() {
                return to_fbs(fbb, *get_station(sched, c.station_id_));
              });
          return CreateParkingEdgeCosts(
              fbb, st, c.car_duration_, c.car_distance_, c.foot_duration_,
              c.foot_distance_, c.foot_accessibility_, c.total_duration_);
        }));
  };
  return utl::to_vec(pes, [&](parking_edges const& pe) {
    return CreateParkingEdge(fbb, create_parking(fbb, pe.parking_),
                             pe.uses_car(), create_costs(pe.outward_costs_),
                             create_costs(pe.return_costs_));
  });
}

msg_ptr make_osrm_via_request(geo::latlng const& start,
                              geo::latlng const& dest) {
  auto const waypoints =
      std::vector<Position>{to_position(start), to_position(dest)};

  message_creator mc;
  mc.create_and_finish(
      MsgContent_OSRMViaRouteRequest,
      CreateOSRMViaRouteRequest(mc, mc.CreateString("car"),
                                mc.CreateVectorOfStructs(waypoints))
          .Union(),
      "/osrm/via");
  return make_msg(mc);
}

Offset<Route> find_matching_ppr_route(FootRoutingResponse const* ppr_resp,
                                      int const target_duration,
                                      int const target_accessibility,
                                      message_creator& fbb) {
  Offset<Route> ppr_route = 0;
  if (ppr_resp->routes()->size() == 1) {
    Route const* best_match = nullptr;

    auto const routes = ppr_resp->routes()->Get(0);
    for (Route const* route : *routes->routes()) {
      auto const accessibility_diff =
          abs(static_cast<int>(route->accessibility()) - target_accessibility);
      auto const duration_diff =
          abs(static_cast<int>(route->duration()) - target_duration);
      if (best_match == nullptr ||
          accessibility_diff <
              abs(static_cast<int>(best_match->accessibility()) -
                  target_accessibility) ||
          (accessibility_diff ==
               abs(static_cast<int>(best_match->accessibility()) -
                   target_accessibility) &&
           duration_diff < abs(static_cast<int>(best_match->duration()) -
                               target_duration))) {
        best_match = route;
      }
    }

    if (best_match != nullptr) {
      auto const route = best_match;
      // ppr_route = motis_copy_table(Route, fbb, route); // <- crashes
      ppr_route = CreateRoute(
          fbb, route->distance(), route->duration(), route->duration_exact(),
          route->duration_division(), route->accessibility(),
          route->accessibility_exact(), route->accessibility_division(),
          route->start(), route->destination(),
          fbb.CreateVector(utl::to_vec(*route->steps(),
                                       [&](RouteStep const* step) {
                                         return motis_copy_table(RouteStep, fbb,
                                                                 step);
                                       })),
          fbb.CreateVector(utl::to_vec(*route->edges(),
                                       [&](Edge const* edge) {
                                         return motis_copy_table(Edge, fbb,
                                                                 edge);
                                       })),
          CreatePolyline(fbb, fbb.CreateVector(utl::to_vec(
                                  *route->path()->coordinates(),
                                  [](auto const& c) { return c; }))),
          route->elevation_up(), route->elevation_down());
    }
  }
  return ppr_route;
}

struct parking::impl {
  explicit impl(std::string const& parking_file,
                std::string const& footedges_db_file, std::size_t db_max_size)
      : parkings_(parking_file), db_(footedges_db_file, db_max_size, true) {}

  void update_ppr_profiles() { ppr_profiles_.update(); }

  msg_ptr geo_lookup(msg_ptr const& msg) {
    auto const req = motis_content(ParkingGeoRequest, msg);
    message_creator fbb;
    fbb.create_and_finish(
        MsgContent_ParkingGeoResponse,
        CreateParkingGeoResponse(
            fbb,
            fbb.CreateVector(utl::to_vec(
                parkings_.get_parkings(to_latlng(req->pos()), req->radius()),
                [&](auto const& p) { return create_parking(fbb, p); })))
            .Union());
    return make_msg(fbb);
  }

  msg_ptr id_lookup(msg_ptr const& msg) {
    auto const req = motis_content(ParkingLookupRequest, msg);
    auto const p = parkings_.get_parking(req->id());
    if (p) {
      Position pos{p->location_.lat_, p->location_.lng_};
      message_creator fbb;
      fbb.create_and_finish(
          MsgContent_ParkingLookupResponse,
          CreateParkingLookupResponse(fbb, create_parking(fbb, *p)).Union());
      return make_msg(fbb);
    } else {
      throw std::system_error(error::invalid_parking_id);
    }
  }

  msg_ptr parking_edge(msg_ptr const& msg) {
    auto const req = motis_content(ParkingEdgeRequest, msg);
    auto const p = parkings_.get_parking(req->id());
    if (p) {
      Position parking_pos{p->location_.lat_, p->location_.lng_};
      message_creator fbb;

      auto const outward = req->direction() == ParkingEdgeDirection_Outward;
      auto const car_start = outward ? to_latlng(req->start()) : p->location_;
      auto const car_dest =
          outward ? p->location_ : to_latlng(req->destination());
      auto const foot_start = outward ? p->location_ : to_latlng(req->start());
      auto const foot_dest = outward ? *req->destination() : parking_pos;

      auto const osrm_msg =
          motis_call(make_osrm_via_request(car_start, car_dest))->val();
      auto const osrm_resp = motis_content(OSRMViaRouteResponse, osrm_msg);

      auto const ppr_req = make_ppr_request(
          foot_start, {foot_dest}, req->ppr_search_options(),
          motis::ppr::SearchDirection_Forward, req->include_steps(),
          req->include_edges(), req->include_path());
      auto const ppr_msg = motis_call(ppr_req)->val();
      auto const ppr_resp = motis_content(FootRoutingResponse, ppr_msg);

      auto const target_duration =
          static_cast<int>(req->duration() - osrm_resp->time());
      auto const target_accessibility = static_cast<int>(req->accessibility());
      auto const ppr_route = find_matching_ppr_route(ppr_resp, target_duration,
                                                     target_accessibility, fbb);

      fbb.create_and_finish(
          MsgContent_ParkingEdgeResponse,
          CreateParkingEdgeResponse(
              fbb, create_parking(fbb, *p),
              motis_copy_table(OSRMViaRouteResponse, fbb, osrm_resp), ppr_route,
              true)
              .Union());
      return make_msg(fbb);
    } else if (req->id() == 0) {
      // parking edge without car, i.e. foot only without any parking
      message_creator fbb;

      auto const foot_start = to_latlng(req->start());
      auto const foot_dest = *req->destination();

      auto const ppr_req = make_ppr_request(
          foot_start, {foot_dest}, req->ppr_search_options(),
          motis::ppr::SearchDirection_Forward, req->include_steps(),
          req->include_edges(), req->include_path());
      auto const ppr_msg = motis_call(ppr_req)->val();
      auto const ppr_resp = motis_content(FootRoutingResponse, ppr_msg);

      auto const target_duration = static_cast<int>(req->duration());
      auto const target_accessibility = static_cast<int>(req->accessibility());
      auto const ppr_route = find_matching_ppr_route(ppr_resp, target_duration,
                                                     target_accessibility, fbb);

      Position parking_pos{0, 0};
      fbb.create_and_finish(
          MsgContent_ParkingEdgeResponse,
          CreateParkingEdgeResponse(
              fbb, create_parking(fbb, *p),
              CreateOSRMViaRouteResponse(
                  fbb, 0, 0,
                  CreatePolyline(fbb, fbb.CreateVector(std::vector<double>{}))),
              ppr_route, false)
              .Union());
      return make_msg(fbb);
    } else {
      throw std::system_error(error::invalid_parking_id);
    }
  }

  msg_ptr parking_edges_req(msg_ptr const& msg) {
    auto const req = motis_content(ParkingEdgesRequest, msg);
    auto const pos = to_latlng(req->pos());

    int64_t get_parkings_duration = 0L;
    int64_t parking_edges_duration = 0L;
    int64_t nocar_parking_edges_duration = 0L;
    uint64_t parking_count = 0UL;
    uint64_t parking_edge_count = 0UL;
    uint64_t nocar_parking_edge_count = 0UL;
    parking_edge_stats pe_stats{};

    MOTIS_START_TIMING(get_parkings_timing);
    auto const start_parking_radius = req->max_car_duration() * CAR_SPEED;
    auto parkings = parkings_.get_parkings(pos, start_parking_radius);
    MOTIS_STOP_TIMING(get_parkings_timing);
    get_parkings_duration = MOTIS_TIMING_MS(get_parkings_timing);
    parking_count = parkings.size();

    MOTIS_START_TIMING(parking_edges_timing);
    auto const walking_speed = ppr_profiles_.get_walking_speed(
        req->ppr_search_options()->profile()->str());
    auto edges = get_parking_edges(
        parkings, pos, req->filtered_stations(), req->max_car_duration(),
        req->ppr_search_options(), db_, pe_stats, req->include_outward(),
        req->include_return(), walking_speed);
    MOTIS_STOP_TIMING(parking_edges_timing);
    parking_edges_duration = MOTIS_TIMING_MS(parking_edges_timing);
    parking_edge_count = edges.size();

    MOTIS_START_TIMING(nocar_parking_edges_timing);
    if (req->include_without_car()) {
      nocar_parking_edge_count = add_nocar_parking_edges(
          edges, pos, req->filtered_stations(), req->ppr_search_options(),
          pe_stats, req->include_outward(), req->include_return(),
          walking_speed);
    }
    MOTIS_STOP_TIMING(nocar_parking_edges_timing);
    nocar_parking_edges_duration = MOTIS_TIMING_MS(nocar_parking_edges_timing);

    message_creator fbb;
    fbb.create_and_finish(
        MsgContent_ParkingEdgesResponse,
        CreateParkingEdgesResponse(
            fbb, fbb.CreateVector(create_parking_edges(fbb, edges)),
            to_fbs(fbb, "parking.parking_edges",
                   {{"osrm_duration",
                     static_cast<uint64_t>(pe_stats.osrm_duration_)},
                    {"parking_edge_duration",
                     static_cast<uint64_t>(pe_stats.parking_edge_duration_)},
                    {"parking_ppr_duration",
                     static_cast<uint64_t>(pe_stats.parking_ppr_duration_)},
                    {"nocar_ppr_duration",
                     static_cast<uint64_t>(pe_stats.nocar_ppr_duration_)},
                    {"get_parkings_duration",
                     static_cast<uint64_t>(get_parkings_duration)},
                    {"parking_edges_duration",
                     static_cast<uint64_t>(parking_edges_duration)},
                    {"nocar_parking_edges_duration",
                     static_cast<uint64_t>(nocar_parking_edges_duration)},
                    {"parking_count", parking_count},
                    {"parking_edge_count", parking_edge_count},
                    {"nocar_parking_edge_count", nocar_parking_edge_count}}))
            .Union());
    return make_msg(fbb);
  }

private:
  parkings parkings_;
  database db_;
  ppr_profiles ppr_profiles_;
};

parking::parking() : module("Parking", "parking") {
  param(db_max_size_, "db_max_size", "virtual memory map size");
  param(max_walk_duration_, "max_walk_duration", "max walk duration (minutes)");
  param(edge_rtree_max_size_, "import.edge_rtree_max_size",
        "Maximum size for ppr edge r-tree file");
  param(area_rtree_max_size_, "import.area_rtree_max_size",
        "Maximum size for ppr area r-tree file");
  param(lock_rtrees_, "import.lock_rtrees", "Lock ppr r-trees in memory");
}

parking::~parking() = default;

fs::path parking::module_data_dir() const {
  return get_data_directory() / "parking";
}

std::string parking::parking_file() const {
  return (module_data_dir() / "parking.txt").generic_string();
}

std::string parking::footedges_db_file() const {
  return (module_data_dir() / "parking_footedges.db").generic_string();
}

std::string parking::stations_per_parking_file() const {
  return (module_data_dir() / "stations_per_parking.txt").generic_string();
}

void parking::import(registry& reg) {
  std::make_shared<event_collector>(
      get_data_directory().generic_string(), "parking", reg,
      [this](std::map<std::string, msg_ptr> const& dependencies) {
        using namespace ::motis::parking::prepare;
        using import::OSMEvent;
        using import::PPREvent;

        auto const dir = get_data_directory() / "parking";
        auto const osm_ev = motis_content(OSMEvent, dependencies.at("OSM"));
        auto const ppr_ev = motis_content(PPREvent, dependencies.at("PPR"));
        auto const state = import_state{data_path(osm_ev->path()->str()),
                                        osm_ev->hash(),
                                        osm_ev->size(),
                                        data_path(ppr_ev->graph_path()->str()),
                                        ppr_ev->graph_hash(),
                                        ppr_ev->graph_size(),
                                        ppr_ev->profiles_hash(),
                                        max_walk_duration_,
                                        get_schedule().hash_};

        if (read_ini<import_state>(dir / "import.ini") != state) {
          fs::create_directories(dir);

          auto ppr_profiles =
              std::map<std::string, ::motis::ppr::profile_info>{};
          ::motis::ppr::read_profile_files(
              utl::to_vec(*ppr_ev->profiles(),
                          [](auto const& p) { return p->path()->str(); }),
              ppr_profiles);
          for (auto& p : ppr_profiles) {
            p.second.profile_.duration_limit_ = max_walk_duration_ * 60;
          }

          std::vector<motis::parking::parking_lot> parking_data;

          auto progress_tracker = utl::get_active_progress_tracker();
          progress_tracker->status("Extract Parkings");

          utl::verify(extract_parkings(osm_ev->path()->str(), parking_file(),
                                       parking_data),
                      "Parking data extraction failed");

          auto park = parkings{std::move(parking_data)};
          auto st = stations{get_schedule()};

          progress_tracker->status("Compute Foot Edges");
          compute_foot_edges(st, park, footedges_db_file(),
                             ppr_ev->graph_path()->str(), edge_rtree_max_size_,
                             area_rtree_max_size_, lock_rtrees_, ppr_profiles,
                             std::thread::hardware_concurrency(),
                             stations_per_parking_file());

          write_ini(dir / "import.ini", state);
        }

        import_successful_ = true;
      })
      ->require("OSM",
                [](msg_ptr const& msg) {
                  return msg->get()->content_type() == MsgContent_OSMEvent;
                })
      ->require("SCHEDULE",
                [](msg_ptr const& msg) {
                  return msg->get()->content_type() == MsgContent_ScheduleEvent;
                })
      ->require("PPR", [](msg_ptr const& msg) {
        return msg->get()->content_type() == MsgContent_PPREvent;
      });
}

void parking::init(motis::module::registry& reg) {
  try {
    impl_ = std::make_unique<impl>(parking_file(), footedges_db_file(),
                                   db_max_size_);

    reg.register_op("/parking/geo",
                    [this](auto&& m) { return impl_->geo_lookup(m); });
    reg.register_op("/parking/lookup",
                    [this](auto&& m) { return impl_->id_lookup(m); });
    reg.register_op("/parking/edge",
                    [this](auto&& m) { return impl_->parking_edge(m); });
    reg.register_op("/parking/edges",
                    [this](auto&& m) { return impl_->parking_edges_req(m); });
    reg.subscribe("/init", [this]() { impl_->update_ppr_profiles(); });
  } catch (std::exception const& e) {
    LOG(logging::warn) << "parking module not initialized (" << e.what() << ")";
  }
}

}  // namespace motis::parking
