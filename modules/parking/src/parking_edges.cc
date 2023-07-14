#include "motis/parking/parking_edges.h"

#include <algorithm>
#include <mutex>

#include "utl/to_vec.h"

#include "ppr/routing/search_profile.h"

#include "motis/core/common/timing.h"
#include "motis/core/access/station_access.h"
#include "motis/module/context/motis_call.h"
#include "motis/module/context/motis_parallel_for.h"

#include "motis/parking/database.h"
#include "motis/parking/mumo_edges.h"

using namespace motis::osrm;
using namespace motis::lookup;
using namespace motis::module;
using namespace motis::ppr;
using namespace flatbuffers;
using namespace ppr::routing;

namespace motis::parking {

std::vector<Station const*> remove_dest_stations(
    Vector<Offset<Station>> const* start_stations,
    Vector<Offset<Station>> const* dest_stations) {
  std::vector<Station const*> result;
  for (auto const& s : *start_stations) {
    if (!std::any_of(dest_stations->begin(), dest_stations->end(),
                     [&](auto const& ds) {
                       return s->id()->str() == ds->id()->str();
                     })) {
      result.emplace_back(s);
    }
  }
  return result;
}

struct osrm_parking_response {
  msg_ptr outward_msg_{};
  msg_ptr return_msg_{};
  OSRMOneToManyResponse const* outward_resp_{};
  OSRMOneToManyResponse const* return_resp_{};
};

osrm_parking_response route_osrm(geo::latlng const& start_pos,
                                 std::vector<parking_lot> const& parkings,
                                 bool include_outward, bool include_return) {
  osrm_parking_response res{};
  future outward_req, return_req;

  if (include_outward) {
    outward_req = motis_call(
        make_osrm_request(start_pos, parkings, "car", SearchDir_Forward));
  }
  if (include_return) {
    return_req = motis_call(
        make_osrm_request(start_pos, parkings, "car", SearchDir_Backward));
  }

  if (include_outward) {
    res.outward_msg_ = outward_req->val();
    res.outward_resp_ = motis_content(OSRMOneToManyResponse, res.outward_msg_);
    assert(res.outward_resp_->costs()->size() == parkings.size());
  }
  if (include_return) {
    res.return_msg_ = return_req->val();
    res.return_resp_ = motis_content(OSRMOneToManyResponse, res.return_msg_);
    assert(res.return_resp_->costs()->size() == parkings.size());
  }

  return res;
}

struct ppr_parking_response {
  msg_ptr outward_msg_{};
  msg_ptr return_msg_{};
  FootRoutingResponse const* outward_resp_{};
  FootRoutingResponse const* return_resp_{};
};

ppr_parking_response route_ppr(geo::latlng const& parking_pos,
                               std::vector<Station const*> const& stations,
                               SearchOptions const* ppr_search_options,
                               bool include_outward, bool include_return) {
  ppr_parking_response res{};
  future outward_req, return_req;

  auto const station_positions = utl::to_vec(
      stations, [&](Station const* station) { return *station->pos(); });

  if (include_outward) {
    outward_req = motis_call(make_ppr_request(
        parking_pos, station_positions, ppr_search_options, SearchDir_Forward));
  }

  if (include_return) {
    return_req =
        motis_call(make_ppr_request(parking_pos, station_positions,
                                    ppr_search_options, SearchDir_Backward));
  }

  if (include_outward) {
    res.outward_msg_ = outward_req->val();
    res.outward_resp_ = motis_content(FootRoutingResponse, res.outward_msg_);
    assert(res.outward_resp_->routes()->size() == stations.size());
  }
  if (include_return) {
    res.return_msg_ = return_req->val();
    res.return_resp_ = motis_content(FootRoutingResponse, res.return_msg_);
    assert(res.return_resp_->routes()->size() == stations.size());
  }

  return res;
}

void add_parking_edges(std::vector<parking_edges>& edges,
                       parking_lot const& parking,
                       Vector<Offset<Station>> const* dest_stations,
                       motis::ppr::SearchOptions const* ppr_search_options,
                       Cost const* osrm_outward_costs,
                       Cost const* osrm_return_costs, std::mutex& mutex,
                       parking_edge_stats& pe_stats, bool include_outward,
                       bool include_return, double walking_speed) {
  auto const parking_station_radius =
      ppr_search_options->duration_limit() * walking_speed;
  auto const geo_msg =
      motis_call(
          make_geo_station_request(parking.location_, parking_station_radius))
          ->val();
  auto const geo_resp = motis_content(LookupGeoStationResponse, geo_msg);
  auto const stations =
      remove_dest_stations(geo_resp->stations(), dest_stations);

  MOTIS_START_TIMING(ppr_timing);
  auto const ppr_res =
      route_ppr(parking.location_, stations, ppr_search_options,
                include_outward, include_return);
  MOTIS_STOP_TIMING(ppr_timing);

  std::vector<parking_edge_costs> outward_costs;
  std::vector<parking_edge_costs> return_costs;

  for (auto i = 0UL; i < stations.size(); ++i) {
    auto const station = stations[i];
    if (include_outward) {
      for (auto const& r : *ppr_res.outward_resp_->routes()->Get(i)->routes()) {
        parking_edge_costs const c{station, osrm_outward_costs, r};
        if (c.valid()) {
          outward_costs.emplace_back(c);
        }
      }
    }
    if (include_return) {
      for (auto const& r : *ppr_res.return_resp_->routes()->Get(i)->routes()) {
        parking_edge_costs const c{station, osrm_return_costs, r};
        if (c.valid()) {
          return_costs.emplace_back(c);
        }
      }
    }
  }

  std::lock_guard<std::mutex> const guard{mutex};
  pe_stats.parking_ppr_duration_ += MOTIS_TIMING_MS(ppr_timing);

  if ((outward_costs.empty() && return_costs.empty()) ||
      (include_outward && include_return &&
       (outward_costs.empty() || return_costs.empty()))) {
    return;
  }
  edges.emplace_back(parking, outward_costs, return_costs);
}

std::vector<parking_edges> get_custom_parking_edges(
    std::vector<parking_lot> const& parkings, geo::latlng const& start_pos,
    Vector<Offset<Station>> const* dest_stations, int max_car_duration,
    motis::ppr::SearchOptions const* ppr_search_options,
    parking_edge_stats& pe_stats, bool include_outward, bool include_return,
    double walking_speed) {
  std::vector<parking_edges> edges;

  MOTIS_START_TIMING(osrm_timing);
  auto const osrm_res =
      route_osrm(start_pos, parkings, include_outward, include_return);
  MOTIS_STOP_TIMING(osrm_timing);
  pe_stats.osrm_duration_ = MOTIS_TIMING_MS(osrm_timing);

  std::mutex mutex;

  MOTIS_START_TIMING(pe_timing);
  std::vector<std::size_t> parking_ids;
  parking_ids.reserve(parkings.size());
  for (auto parking_idx = 0UL; parking_idx < parkings.size(); ++parking_idx) {
    parking_ids.emplace_back(parking_idx);
  }

  motis_parallel_for(parking_ids, [&](auto const parking_idx) {
    auto const& parking = parkings[parking_idx];
    auto const osrm_outward_costs =
        include_outward ? osrm_res.outward_resp_->costs()->Get(parking_idx)
                        : nullptr;
    auto const osrm_return_costs =
        include_return ? osrm_res.return_resp_->costs()->Get(parking_idx)
                       : nullptr;
    if ((osrm_outward_costs != nullptr &&
         osrm_outward_costs->duration() > max_car_duration) ||
        (osrm_return_costs != nullptr &&
         osrm_return_costs->duration() > max_car_duration)) {
      return;
    }
    add_parking_edges(edges, parking, dest_stations, ppr_search_options,
                      osrm_outward_costs, osrm_return_costs, mutex, pe_stats,
                      include_outward, include_return, walking_speed);
  });
  MOTIS_STOP_TIMING(pe_timing);

  pe_stats.parking_edge_duration_ = MOTIS_TIMING_MS(pe_timing);

  return edges;
}

std::vector<parking_edges> get_cached_parking_edges(
    station_lookup const& st, std::vector<parking_lot> const& parkings,
    geo::latlng const& start_pos, Vector<Offset<Station>> const* dest_stations,
    int max_car_duration, motis::ppr::SearchOptions const* ppr_search_options,
    database& db, parking_edge_stats& pe_stats, bool include_outward,
    bool include_return, double walking_speed) {
  std::vector<parking_edges> edges;

  MOTIS_START_TIMING(osrm_timing);
  auto const osrm_res =
      route_osrm(start_pos, parkings, include_outward, include_return);
  MOTIS_STOP_TIMING(osrm_timing);
  pe_stats.osrm_duration_ = MOTIS_TIMING_MS(osrm_timing);

  auto const foot_duration_limit = static_cast<duration>(
      std::ceil(ppr_search_options->duration_limit() / 60));
  auto const filter_station = [&](FootEdge const* fe) {
    auto const sid = fe->station_id()->str();
    return fe->duration() > foot_duration_limit || !st.get(sid).valid() ||
           std::any_of(dest_stations->begin(), dest_stations->end(),
                       [&](auto const& ds) { return sid == ds->id()->str(); });
  };

  std::mutex mutex;
  MOTIS_START_TIMING(pe_timing);
  for (auto parking_idx = 0UL; parking_idx < parkings.size(); ++parking_idx) {
    auto const& parking = parkings[parking_idx];
    auto const osrm_outward_costs =
        include_outward ? osrm_res.outward_resp_->costs()->Get(parking_idx)
                        : nullptr;
    auto const osrm_return_costs =
        include_return ? osrm_res.return_resp_->costs()->Get(parking_idx)
                       : nullptr;
    if ((osrm_outward_costs != nullptr &&
         osrm_outward_costs->duration() > max_car_duration) ||
        (osrm_return_costs != nullptr &&
         osrm_return_costs->duration() > max_car_duration)) {
      continue;
    }

    auto const foot_edges =
        db.get_footedges(parking.id_, ppr_search_options->profile()->str());
    if (foot_edges) {
      std::vector<parking_edge_costs> outward_costs;
      std::vector<parking_edge_costs> return_costs;

      for (auto const& fe : *foot_edges->get()->outward_edges()) {
        if (filter_station(fe)) {
          continue;
        }
        auto const station_id = fe->station_id()->str();
        parking_edge_costs const c{station_id,          "",
                                   osrm_outward_costs,  fe->duration(),
                                   fe->accessibility(), fe->distance()};
        if (c.valid()) {
          outward_costs.emplace_back(c);
        }
      }

      for (auto const& fe : *foot_edges->get()->return_edges()) {
        if (filter_station(fe)) {
          continue;
        }
        auto const station_id = fe->station_id()->str();
        parking_edge_costs const c{station_id,          "",
                                   osrm_return_costs,   fe->duration(),
                                   fe->accessibility(), fe->distance()};
        if (c.valid()) {
          return_costs.emplace_back(c);
        }
      }
      if (!outward_costs.empty() && !return_costs.empty()) {
        edges.emplace_back(parking, outward_costs, return_costs);
      }
    } else {
      add_parking_edges(edges, parking, dest_stations, ppr_search_options,
                        osrm_outward_costs, osrm_return_costs, mutex, pe_stats,
                        include_outward, include_return, walking_speed);
    }
  }
  MOTIS_STOP_TIMING(pe_timing);

  pe_stats.parking_edge_duration_ = MOTIS_TIMING_MS(pe_timing);

  return edges;
}

std::vector<parking_edges> get_parking_edges(
    station_lookup const& st, std::vector<parking_lot> const& parkings,
    geo::latlng const& start_pos, Vector<Offset<Station>> const* dest_stations,
    int max_car_duration, motis::ppr::SearchOptions const* ppr_search_options,
    database& db, parking_edge_stats& pe_stats, bool include_outward,
    bool include_return, double walking_speed) {
  return get_cached_parking_edges(st, parkings, start_pos, dest_stations,
                                  max_car_duration, ppr_search_options, db,
                                  pe_stats, include_outward, include_return,
                                  walking_speed);
  /*
  return get_custom_parking_edges(
      parkings, start_pos, dest_stations, max_car_duration, ppr_search_options,
      pe_stats, include_outward, include_return, walking_speed);
  */
}

unsigned add_nocar_parking_edges(
    std::vector<parking_edges>& edges, geo::latlng const& start_pos,
    Vector<Offset<Station>> const* dest_stations,
    motis::ppr::SearchOptions const* ppr_search_options,
    parking_edge_stats& pe_stats, bool include_outward, bool include_return,
    double walking_speed) {
  auto count = 0U;
  auto const parking_station_radius =
      ppr_search_options->duration_limit() * walking_speed;
  auto const geo_msg =
      motis_call(make_geo_station_request(start_pos, parking_station_radius))
          ->val();
  auto const geo_resp = motis_content(LookupGeoStationResponse, geo_msg);
  auto const stations =
      remove_dest_stations(geo_resp->stations(), dest_stations);

  if (stations.empty()) {
    return count;
  }

  MOTIS_START_TIMING(ppr_timing);
  auto const ppr_res = route_ppr(start_pos, stations, ppr_search_options,
                                 include_outward, include_return);
  MOTIS_STOP_TIMING(ppr_timing);
  pe_stats.nocar_ppr_duration_ = MOTIS_TIMING_MS(ppr_timing);

  std::vector<parking_edge_costs> outward_costs;
  std::vector<parking_edge_costs> return_costs;

  for (auto i = 0UL; i < stations.size(); ++i) {
    auto const station = stations[i];
    if (include_outward) {
      for (auto const& r : *ppr_res.outward_resp_->routes()->Get(i)->routes()) {
        parking_edge_costs const c{station, nullptr, r};
        if (c.valid()) {
          outward_costs.emplace_back(c);
          ++count;
        }
      }
    }
    if (include_return) {
      for (auto const& r : *ppr_res.return_resp_->routes()->Get(i)->routes()) {
        parking_edge_costs const c{station, nullptr, r};
        if (c.valid()) {
          return_costs.emplace_back(c);
          ++count;
        }
      }
    }
  }

  if ((outward_costs.empty() && return_costs.empty()) ||
      (include_outward && include_return &&
       (outward_costs.empty() || return_costs.empty()))) {
    return count;
  }
  parking_lot const no_parking;
  edges.emplace_back(no_parking, outward_costs, return_costs);

  return count;
}

}  // namespace motis::parking
