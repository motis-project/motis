#include "motis/intermodal/mumo_edge.h"

#include <algorithm>

#include "utl/erase_if.h"

#include "motis/core/common/constants.h"
#include "motis/module/context/motis_call.h"
#include "motis/module/message.h"

#include "motis/intermodal/error.h"

using namespace geo;
using namespace flatbuffers;
using namespace motis::routing;
using namespace motis::lookup;
using namespace motis::osrm;
using namespace motis::module;
using namespace motis::ppr;
using namespace motis::parking;

namespace motis::intermodal {

inline geo::latlng to_latlng(Position const* pos) {
  return {pos->lat(), pos->lng()};
}

msg_ptr make_geo_request(latlng const& pos, double radius) {
  Position fbs_position{pos.lat_, pos.lng_};
  message_creator mc;
  mc.create_and_finish(
      MsgContent_LookupGeoStationRequest,
      CreateLookupGeoStationRequest(mc, &fbs_position, 0, radius).Union(),
      "/lookup/geo_station");
  return make_msg(mc);
}

msg_ptr make_osrm_request(latlng const& pos,
                          Vector<Offset<Station>> const* stations,
                          std::string const& profile, SearchDir direction) {
  Position fbs_position{pos.lat_, pos.lng_};
  std::vector<Position> many;
  for (auto const* station : *stations) {
    many.push_back(*station->pos());
  }

  message_creator mc;
  mc.create_and_finish(
      MsgContent_OSRMOneToManyRequest,
      CreateOSRMOneToManyRequest(mc, mc.CreateString(profile), direction,
                                 &fbs_position, mc.CreateVectorOfStructs(many))
          .Union(),
      "/osrm/one_to_many");
  return make_msg(mc);
}

void osrm_edges(latlng const& pos, int max_dur, int max_dist,
                mumo_type const type, SearchDir direction,
                appender_fun const& appender) {
  auto const geo_msg = motis_call(make_geo_request(pos, max_dist))->val();
  auto const geo_resp = motis_content(LookupGeoStationResponse, geo_msg);
  auto const stations = geo_resp->stations();

  auto const osrm_msg =
      motis_call(make_osrm_request(pos, stations, to_string(type), direction))
          ->val();
  auto const osrm_resp = motis_content(OSRMOneToManyResponse, osrm_msg);

  for (auto i = 0UL; i < stations->size(); ++i) {
    auto const dur = osrm_resp->costs()->Get(i)->duration();
    if (dur > max_dur) {
      continue;
    }

    appender(stations->Get(i)->id()->str(), to_latlng(stations->Get(i)->pos()),
             dur / 60, 0, type, 0);
  }
}

msg_ptr make_ppr_request(latlng const& pos,
                         Vector<Offset<Station>> const* stations,
                         SearchOptions const* search_options, SearchDir dir) {
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

void ppr_edges(latlng const& pos, SearchOptions const* search_options,
               SearchDir direction, appender_fun const& appender,
               ppr_profiles const& profiles) {
  auto const max_dur = search_options->duration_limit();
  if (max_dur == 0) {
    return;
  }
  auto const max_dist =
      max_dur * profiles.get_walking_speed(search_options->profile()->str());
  auto const geo_msg = motis_call(make_geo_request(pos, max_dist))->val();
  auto const geo_resp = motis_content(LookupGeoStationResponse, geo_msg);
  auto const stations = geo_resp->stations();

  auto const ppr_msg =
      motis_call(make_ppr_request(pos, stations, search_options, direction))
          ->val();
  auto const ppr_resp = motis_content(FootRoutingResponse, ppr_msg);

  auto const routes = ppr_resp->routes();
  assert(routes->size() <= stations->size());
  for (auto i = 0U; i < routes->size(); ++i) {
    auto const dest_routes = routes->Get(i);
    auto const dest_id = stations->Get(i)->id()->str();
    auto const dest_pos = to_latlng(stations->Get(i)->pos());
    for (auto const& route : *dest_routes->routes()) {
      appender(dest_id, dest_pos, route->duration(), route->accessibility(),
               mumo_type::FOOT, 0);
    }
  }
}

void car_parking_edges(latlng const& pos, int max_car_duration,
                       SearchOptions const* ppr_search_options,
                       SearchDir direction, appender_fun const& appender,
                       mumo_stats_appender_fun const& mumo_stats_appender,
                       std::string const& mumo_stats_prefix) {
  Position fbs_position{pos.lat_, pos.lng_};
  message_creator mc;
  mc.create_and_finish(
      MsgContent_ParkingEdgesRequest,
      CreateParkingEdgesRequest(
          mc, &fbs_position, max_car_duration,
          motis_copy_table(SearchOptions, mc, ppr_search_options),
          mc.CreateVector(std::vector<Offset<Station>>{}),
          direction == SearchDir_Forward, direction == SearchDir_Backward,
          false)
          .Union(),
      "/parking/edges");
  auto const pe_msg = motis_call(make_msg(mc))->val();
  auto const pe_res = motis_content(ParkingEdgesResponse, pe_msg);
  for (auto const& pe : *pe_res->edges()) {
    auto const& costs = direction == SearchDir_Forward ? pe->outward_costs()
                                                       : pe->return_costs();
    for (auto const& c : *costs) {
      auto& e =
          appender(c->station()->id()->str(), to_latlng(c->station()->pos()),
                   c->total_duration(), c->foot_accessibility(),
                   mumo_type::CAR_PARKING, pe->parking()->id());
      e.car_parking_ = {
          pe->parking()->id(),     to_latlng(pe->parking()->pos()),
          c->car_duration(),       c->foot_duration(),
          c->foot_accessibility(), c->total_duration(),
          pe->uses_car()};
    }
  }
  auto stats = from_fbs(pe_res->stats());
  stats.key_ = mumo_stats_prefix + stats.key_;
  mumo_stats_appender(std::move(stats));
}

void gbfs_edges(appender_fun const& appender, SearchDir const dir,
                latlng const& pos,
                std::vector<std::string> const& vehicle_types,
                unsigned const max_walk_duration,
                unsigned const max_ride_duration) {
  using gbfs::GBFSRoutingResponse;

  Position fbs_position{pos.lat_, pos.lng_};
  message_creator mc;
  mc.create_and_finish(
      MsgContent_GBFSRoutingRequest,
      gbfs::CreateGBFSRoutingRequest(
          mc, dir, &fbs_position,
          mc.CreateVector(utl::to_vec(
              vehicle_types, [&](auto&& s) { return mc.CreateString(s); })),
          max_walk_duration, max_ride_duration)
          .Union(),
      "/gbfs/route");
  auto const res_msg = motis_call(make_msg(mc))->val();
  auto const gbfs_res = motis_content(GBFSRoutingResponse, res_msg);
  for (auto const& r : *gbfs_res->routes()) {
    appender(r->station()->id()->str(), to_latlng(r->station()->pos()),
             r->total_duration(), 0, mumo_type::GBFS, 0);
  }
}

void make_edges(Vector<Offset<ModeWrapper>> const* modes, latlng const& pos,
                SearchDir const search_dir, appender_fun const& appender,
                mumo_stats_appender_fun const& mumo_stats_appender,
                std::string const& mumo_stats_prefix,
                ppr_profiles const& profiles) {
  for (auto const& wrapper : *modes) {
    switch (wrapper->mode_type()) {
      case Mode_Foot: {
        auto max_dur =
            reinterpret_cast<Foot const*>(wrapper->mode())->max_duration();
        auto max_dist = max_dur * WALK_SPEED;
        osrm_edges(pos, max_dur, max_dist, mumo_type::FOOT, search_dir,
                   appender);
        break;
      }

      case Mode_Bike: {
        auto max_dur =
            reinterpret_cast<Bike const*>(wrapper->mode())->max_duration();
        auto max_dist = max_dur * BIKE_SPEED;
        osrm_edges(pos, max_dur, max_dist, mumo_type::BIKE, search_dir,
                   appender);
        break;
      }

      case Mode_Car: {
        auto max_dur =
            reinterpret_cast<Car const*>(wrapper->mode())->max_duration();
        auto max_dist = max_dur * CAR_SPEED;
        osrm_edges(pos, max_dur, max_dist, mumo_type::CAR, search_dir,
                   appender);
        break;
      }

      case Mode_FootPPR: {
        auto const options =
            reinterpret_cast<FootPPR const*>(wrapper->mode())->search_options();
        ppr_edges(pos, options, search_dir, appender, profiles);
        break;
      }

      case Mode_CarParking: {
        auto const cp = reinterpret_cast<CarParking const*>(wrapper->mode());
        car_parking_edges(pos, cp->max_car_duration(), cp->ppr_search_options(),
                          search_dir, appender, mumo_stats_appender,
                          mumo_stats_prefix);
        break;
      }

      case Mode_GBFS: {
        auto const gbfs = reinterpret_cast<GBFS const*>(wrapper->mode());
        gbfs_edges(
            appender, search_dir, pos,
            utl::to_vec(*gbfs->vehicle_types(),
                        [](flatbuffers::String const* t) { return t->str(); }),
            gbfs->max_walk_duration() / 60.0, gbfs->max_ride_duration() / 60.0);
        break;
      }

      default: throw std::system_error(error::unknown_mode);
    }
  }
}

void make_starts(IntermodalRoutingRequest const* req, latlng const& pos,
                 appender_fun const& appender,
                 mumo_stats_appender_fun const& mumo_stats_appender,
                 ppr_profiles const& profiles) {
  make_edges(req->start_modes(), pos, SearchDir_Forward, appender,
             mumo_stats_appender, "intermodal.start.", profiles);
}

void make_dests(IntermodalRoutingRequest const* req, latlng const& pos,
                appender_fun const& appender,
                mumo_stats_appender_fun const& mumo_stats_appender,
                ppr_profiles const& profiles) {
  make_edges(req->destination_modes(), pos, SearchDir_Backward, appender,
             mumo_stats_appender, "intermodal.dest.", profiles);
}

void remove_intersection(std::vector<mumo_edge>& starts,
                         std::vector<mumo_edge>& destinations,
                         geo::latlng const& query_start,
                         geo::latlng const& query_destination,
                         SearchDir const dir) {
  if (starts.empty() || destinations.empty()) {
    return;
  }
  if (dir == SearchDir_Forward) {
    utl::erase_if(starts, [&](auto const& start) {
      return std::find_if(begin(destinations), end(destinations),
                          [&](auto const& dest) {
                            return start.to_ == dest.from_;
                          }) != end(destinations) &&
             distance(start.to_pos_, query_destination) <
                 distance(start.to_pos_, query_start);
    });
    utl::erase_if(destinations, [&](auto const& dest) {
      return std::find_if(begin(starts), end(starts),
                          [&](auto const& start) {
                            return start.to_ == dest.from_;
                          }) != end(starts) &&
             distance(dest.from_pos_, query_start) <
                 distance(dest.from_pos_, query_destination);
    });
  } else {
    utl::erase_if(starts, [&](auto const& start) {
      return std::find_if(begin(destinations), end(destinations),
                          [&](auto const& dest) {
                            return start.from_ == dest.to_;
                          }) != end(destinations) &&
             distance(start.from_pos_, query_destination) <
                 distance(start.from_pos_, query_start);
    });
    utl::erase_if(destinations, [&](auto const& dest) {
      return std::find_if(begin(starts), end(starts),
                          [&](auto const& start) {
                            return start.from_ == dest.to_;
                          }) != end(starts) &&
             distance(dest.to_pos_, query_start) <
                 distance(dest.to_pos_, query_destination);
    });
  }
}

std::vector<Offset<AdditionalEdgeWrapper>> write_edges(
    FlatBufferBuilder& fbb, std::vector<mumo_edge> const& starts,
    std::vector<mumo_edge> const& destinations,
    std::vector<mumo_edge const*>& edge_mapping) {
  std::vector<Offset<AdditionalEdgeWrapper>> edges;
  edges.reserve(starts.size() + destinations.size());

  for (auto const& edge : starts) {
    auto const edge_id = static_cast<int>(edge_mapping.size());
    edge_mapping.emplace_back(&edge);
    edges.emplace_back(CreateAdditionalEdgeWrapper(
        fbb, AdditionalEdge_MumoEdge,
        CreateMumoEdge(fbb, fbb.CreateString(edge.from_),
                       fbb.CreateString(edge.to_), edge.duration_, 0,
                       edge.accessibility_, edge_id)
            .Union()));
  }

  for (auto const& edge : destinations) {
    auto const edge_id = static_cast<int>(edge_mapping.size());
    edge_mapping.emplace_back(&edge);
    edges.emplace_back(CreateAdditionalEdgeWrapper(
        fbb, AdditionalEdge_MumoEdge,
        CreateMumoEdge(fbb, fbb.CreateString(edge.from_),
                       fbb.CreateString(edge.to_), edge.duration_, 0,
                       edge.accessibility_, edge_id)
            .Union()));
  }

  return edges;
}

}  // namespace motis::intermodal
