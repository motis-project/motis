#include "motis/intermodal/mumo_edge.h"

#include <algorithm>

#include "utl/erase_if.h"

#include "motis/core/common/constants.h"
#include "motis/module/context/motis_call.h"
#include "motis/module/message.h"

#include "motis/intermodal/direct_connections.h"
#include "motis/intermodal/error.h"
#include "motis/intermodal/ridesharing_edges.h"

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
                          std::string const& profile, Direction direction) {
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
                mumo_type const type, Direction direction,
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
    if (dur > max_dur || dur <= 0) {
      continue;
    }

    appender(stations->Get(i)->id()->str(), to_latlng(stations->Get(i)->pos()),
             dur / 60, 0, type, 0);
  }
}

msg_ptr make_ppr_request(latlng const& pos,
                         Vector<Offset<Station>> const* stations,
                         SearchOptions const* search_options,
                         Direction direction) {
  assert(search_options != nullptr);
  Position const fbs_position{pos.lat_, pos.lng_};

  auto const dir = direction == Direction_Forward ? SearchDirection_Forward
                                                  : SearchDirection_Backward;

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
               Direction direction, appender_fun const& appender,
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
                       Direction direction, appender_fun const& appender,
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
          direction == Direction_Forward, direction == Direction_Backward,
          false)
          .Union(),
      "/parking/edges");
  auto const pe_msg = motis_call(make_msg(mc))->val();
  auto const pe_res = motis_content(ParkingEdgesResponse, pe_msg);
  for (auto const& pe : *pe_res->edges()) {
    auto const& costs = direction == Direction_Forward ? pe->outward_costs()
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

void add_ridesharing_edge(std::vector<mumo_edge>& add_edges_vec,
                          ridesharing::RidesharingEdge const* e) {
  auto& me = add_edges_vec.emplace_back(
      e->from_station_id()->str(), e->to_station_id()->str(),
      to_latlng(e->from_pos()), to_latlng(e->to_pos()),
      (e->rs_duration() + e->ppr_duration()) / 60, e->ppr_accessibility(),
      mumo_type::RIDESHARING, e->parking_id());
  // LOG(logging::info) << e->to_leg() << " to " << e->from_leg();
  me.ridesharing_ = {e->lift_key()->str(),  e->rs_t(),
                     e->rs_price(),         e->from_leg(),
                     e->to_leg(),           to_latlng(e->from_pos()),
                     to_latlng(e->to_pos())};
  me.car_parking_ = {
      {e->parking_id(), to_latlng(e->parking_pos()),
       static_cast<uint16_t>(e->rs_duration() / 60),
       static_cast<uint16_t>(e->ppr_duration()), e->ppr_accessibility(),
       static_cast<uint16_t>(e->rs_duration() + e->ppr_duration() * 60), true}};
}

void make_ridesharing_request(
    ridesharing_edges& rs_edges, latlng const& start, latlng const& dest,
    bool start_is_intermodal, bool dest_is_intermodal, std::time_t t,
    std::pair<uint16_t, SearchOptions const*> mode_data) {
  using ridesharing::RidesharingResponse;

  message_creator mc;
  Position fbs_start{start.lat_, start.lng_};
  Position fbs_dest{dest.lat_, dest.lng_};

  mc.create_and_finish(
      MsgContent_RidesharingRequest,
      motis::ridesharing::CreateRidesharingRequest(
          mc, &fbs_start, &fbs_dest, t, mode_data.first,
          start_is_intermodal
              ? (dest_is_intermodal ? motis::ridesharing::QUERYMODE_BOTH
                                    : motis::ridesharing::QUERYMODE_START)
              : motis::ridesharing::QUERYMODE_DESTINATION,
          motis_copy_table(SearchOptions, mc, mode_data.second))
          .Union(),
      "/ridesharing/edges");
  auto const res = motis_call(make_msg(mc))->val();
  auto const rs_resp = motis_content(RidesharingResponse, res);

  for (auto const mm_edge : *(rs_resp->deps())) {
    add_ridesharing_edge(rs_edges.deps_, mm_edge);
  }
  for (auto const mm_edge : *(rs_resp->arrs())) {
    add_ridesharing_edge(rs_edges.arrs_, mm_edge);
  }
  for (auto const e : *rs_resp->direct_connections()) {
    rs_edges.direct_connections_.emplace_back(direct_connection{
        mumo_type::RIDESHARING, static_cast<duration>(e->rs_duration() / 60), 0,
        e->rs_price(), e->rs_t(),
        ridesharing_edge{e->lift_key()->str(), e->rs_t(), e->rs_price(),
                         e->from_leg(), e->to_leg(), to_latlng(e->from_pos()),
                         to_latlng(e->to_pos())}});
  }
}

void make_edges(Vector<Offset<ModeWrapper>> const* modes, latlng const& pos,
                Direction const osrm_direction, appender_fun const& appender,
                mumo_stats_appender_fun const& mumo_stats_appender,
                std::string const& mumo_stats_prefix,
                ppr_profiles const& profiles) {
  for (auto const& wrapper : *modes) {
    switch (wrapper->mode_type()) {
      case Mode_Foot: {
        auto max_dur =
            reinterpret_cast<Foot const*>(wrapper->mode())->max_duration();
        auto max_dist = max_dur * WALK_SPEED;
        osrm_edges(pos, max_dur, max_dist, mumo_type::FOOT, osrm_direction,
                   appender);
        break;
      }

      case Mode_Bike: {
        auto max_dur =
            reinterpret_cast<Bike const*>(wrapper->mode())->max_duration();
        auto max_dist = max_dur * BIKE_SPEED;
        osrm_edges(pos, max_dur, max_dist, mumo_type::BIKE, osrm_direction,
                   appender);
        break;
      }

      case Mode_Car: {
        auto max_dur =
            reinterpret_cast<Car const*>(wrapper->mode())->max_duration();
        auto max_dist = max_dur * CAR_SPEED;
        osrm_edges(pos, max_dur, max_dist, mumo_type::CAR, osrm_direction,
                   appender);
        break;
      }

      case Mode_FootPPR: {
        auto const options =
            reinterpret_cast<FootPPR const*>(wrapper->mode())->search_options();
        ppr_edges(pos, options, osrm_direction, appender, profiles);
        break;
      }

      case Mode_CarParking: {
        auto const cp = reinterpret_cast<CarParking const*>(wrapper->mode());
        car_parking_edges(pos, cp->max_car_duration(), cp->ppr_search_options(),
                          osrm_direction, appender, mumo_stats_appender,
                          mumo_stats_prefix);
        break;
      }

      case Mode_Ridesharing: {
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
  make_edges(req->start_modes(), pos, Direction_Forward, appender,
             mumo_stats_appender, "intermodal.start.", profiles);
}

void make_dests(IntermodalRoutingRequest const* req, latlng const& pos,
                appender_fun const& appender,
                mumo_stats_appender_fun const& mumo_stats_appender,
                ppr_profiles const& profiles) {
  make_edges(req->destination_modes(), pos, Direction_Backward, appender,
             mumo_stats_appender, "intermodal.dest.", profiles);
}

void remove_intersection(std::vector<mumo_edge>& starts,
                         std::vector<mumo_edge>& destinations,
                         latlng const& query_start,
                         latlng const& query_destination, SearchDir const dir) {
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
    ridesharing_edges const& rs_edges,
    std::vector<mumo_edge const*>& edge_mapping) {
  std::vector<Offset<AdditionalEdgeWrapper>> edges;
  edges.reserve(starts.size() + destinations.size() + rs_edges.arrs_.size() +
                rs_edges.deps_.size());

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

  for (auto const& edge : rs_edges.arrs_) {
    auto const edge_id = static_cast<int>(edge_mapping.size());
    edge_mapping.emplace_back(&edge);
    auto const in =
        Interval{edge.ridesharing_.value().t_, edge.ridesharing_.value().t_};
    edges.emplace_back(CreateAdditionalEdgeWrapper(
        fbb, AdditionalEdge_TimeDependentMumoEdge,
        CreateTimeDependentMumoEdge(
            fbb,
            CreateMumoEdge(fbb, fbb.CreateString(edge.from_),
                           fbb.CreateString(edge.to_), edge.duration_,
                           edge.ridesharing_.value().price_,
                           edge.accessibility_, edge_id),
            &in)
            .Union()));
  }
  for (auto const& edge : rs_edges.deps_) {
    auto const edge_id = static_cast<int>(edge_mapping.size());

    edge_mapping.emplace_back(&edge);
    auto const in =
        Interval{edge.ridesharing_.value().t_, edge.ridesharing_.value().t_};
    edges.emplace_back(CreateAdditionalEdgeWrapper(
        fbb, AdditionalEdge_TimeDependentMumoEdge,
        CreateTimeDependentMumoEdge(
            fbb,
            CreateMumoEdge(fbb, fbb.CreateString(edge.from_),
                           fbb.CreateString(edge.to_), edge.duration_,
                           edge.ridesharing_.value().price_,
                           edge.accessibility_, edge_id),
            &in)
            .Union()));
  }

  return edges;
}

}  // namespace motis::intermodal
