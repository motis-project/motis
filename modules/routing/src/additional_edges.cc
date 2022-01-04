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

      case AdditionalEdge_NONE: break;
    }
  }
  return edges;
}

}  // namespace motis::routing
