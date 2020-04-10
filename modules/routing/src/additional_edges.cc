#include "motis/routing/additional_edges.h"

#include "motis/core/common/constants.h"
#include "motis/core/schedule/edges.h"
#include "motis/core/schedule/schedule.h"
#include "motis/core/schedule/time.h"
#include "motis/core/access/station_access.h"

namespace fbs = flatbuffers;

namespace motis::routing {

std::vector<edge> create_additional_edges(
    fbs::Vector<fbs::Offset<AdditionalEdgeWrapper>> const* edge_wrappers,
    schedule const& sched) {
  std::vector<edge> edges;
  for (auto const& e : *edge_wrappers) {
    switch (e->additional_edge_type()) {
      case AdditionalEdge_MumoEdge: {
        auto const info =
            reinterpret_cast<MumoEdge const*>(e->additional_edge());
        edges.push_back(make_mumo_edge(
            get_station_node(sched, info->from_station_id()->str()),
            get_station_node(sched, info->to_station_id()->str()),
            info->duration(), info->price(), info->accessibility(),
            info->mumo_id()));
        break;
      }

      case AdditionalEdge_TimeDependentMumoEdge: {
        auto const info = reinterpret_cast<TimeDependentMumoEdge const*>(
            e->additional_edge());
        auto const edge = info->edge();
        edges.push_back(make_time_dependent_mumo_edge(
            get_station_node(sched, edge->from_station_id()->str()),
            get_station_node(sched, edge->to_station_id()->str()),
            edge->duration(), edge->price(), edge->mumo_id(),
            unix_to_motistime(sched.schedule_begin_, info->interval()->begin()),
            unix_to_motistime(sched.schedule_begin_, info->interval()->end())));
        break;
      }

      case AdditionalEdge_PeriodicMumoEdge: {
        auto const info = reinterpret_cast<TimeDependentMumoEdge const*>(
            e->additional_edge());
        auto const edge = info->edge();
        auto const head_station =
            get_station_node(sched, edge->to_station_id()->str());
        edges.push_back(make_periodic_mumo_edge(
            get_station_node(sched, edge->from_station_id()->str()),
            head_station, edge->duration(), edge->price(), edge->mumo_id(),
            info->interval()->begin(), info->interval()->end()));
        break;
      }

      case AdditionalEdge_HotelEdge: {
        auto const info =
            reinterpret_cast<HotelEdge const*>(e->additional_edge());
        auto const edge = info->edge();
        edges.push_back(make_hotel_edge(
            get_station_node(sched, edge->from_station_id()->str()),
            info->earliest_checkout_time(), info->min_stay_duration(),
            edge->price(), edge->mumo_id()));
        break;
      }

      case AdditionalEdge_NONE: break;
    }
  }
  return edges;
}

}  // namespace motis::routing
