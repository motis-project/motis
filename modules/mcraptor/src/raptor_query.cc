#include "motis/mcraptor/raptor_query.h"

#include "motis/core/access/error.h"
#include "motis/core/access/station_access.h"

#include "utl/verify.h"

namespace motis::mcraptor {

using namespace motis::routing;

base_query get_base_query(RoutingRequest const* routing_request,
                          schedule const& sched,
                          raptor_meta_info const& meta_info) {
  //utl::verify(routing_request->search_dir() == SearchDir_Forward,
  //            "RAPTOR currently only supports departure queries");
  utl::verify(routing_request->schedule() == 0U,
              "RAPTOR currently only supports the default schedule");
  base_query q;

  auto const destination_station = routing_request->destination();
  auto const target_eva = destination_station->id()->str();
  std::string start_eva;

  switch (routing_request->start_type()) {
    case Start::Start_PretripStart: {
      auto const* pretrip_start =
          static_cast<PretripStart const*>(routing_request->start());

      auto const start_station = pretrip_start->station();
      start_eva = start_station->id()->str();

      auto const interval = pretrip_start->interval();
      auto const departure_time_begin = interval->begin();
      auto const departure_time_end = interval->end();

      q.source_time_begin_ =
          unix_to_motistime(sched.schedule_begin_, departure_time_begin);
      q.source_time_end_ =
          unix_to_motistime(sched.schedule_begin_, departure_time_end);
    } break;

    case Start::Start_OntripStationStart: {
      auto const* ontrip_start =
          static_cast<OntripStationStart const*>(routing_request->start());

      auto const start_station = ontrip_start->station();
      start_eva = start_station->id()->str();

      auto const departure_time = ontrip_start->departure_time();
      q.source_time_begin_ =
          unix_to_motistime(sched.schedule_begin_, departure_time);
      q.source_time_end_ = q.source_time_begin_;

    } break;

    case Start::Start_OntripTrainStart: {
      throw std::system_error(access::error::not_implemented);
    }

    default: {
      throw utl::fail("No valid start found in raptor query");
    }
  }

  q.source_ = meta_info.eva_to_raptor_id_.at(start_eva);
  q.target_ = meta_info.eva_to_raptor_id_.at(target_eva);

  q.forward_ = (routing_request->search_dir() == SearchDir::SearchDir_Forward);
  q.ontrip_ = routing_request->start_type() != Start::Start_PretripStart;

  if (!q.forward_) {
    stop_id t = q.source_;
    q.source_ = q.target_;
    q.target_ = t;
  }

  // Don't use start meta stations in the ontrip case, because:
  // do as the routing module does
  q.use_start_metas_ =
      routing_request->use_start_metas() &&
      routing_request->start_type() != Start_OntripStationStart;
  q.use_dest_metas_ = routing_request->use_dest_metas();

  q.use_start_footpaths_ = routing_request->use_start_footpaths();

//  q.query_edges_ = create_additional_edges(routing_request->additional_edges(), sched);
  std::vector<raptor_edge> edges;
  for (auto const& e : *routing_request->additional_edges()) {
    switch (e->additional_edge_type()) {
      case AdditionalEdge_MumoEdge: {
        auto const info =
            reinterpret_cast<MumoEdge const*>(e->additional_edge());
        auto from = info->from_station_id()->str();
        auto to = info->to_station_id()->str();
        if(!q.forward_) {
          auto t = from;
          from = to;
          to = t;
        }
        edges.push_back(raptor_edge{info->duration(),
                                             meta_info.eva_to_raptor_id_.at(from),
                                             meta_info.eva_to_raptor_id_.at(to)});
        break;
      }
    }
  }
  for(raptor_edge edge : edges){
    if(edge.from_ == 0) {
      q.raptor_edges_start_.push_back(edge);
    }
    else if(edge.to_ == 1) {
      q.raptor_edges_end_.push_back(edge);
    }
  }

  if(q.target_ == 1) {
    for(raptor_edge edge : q.raptor_edges_end_) {
      q.targets_.push_back(edge.from_);
    }
  }
  else {
    q.raptor_edges_end_.push_back(raptor_edge{0, q.target_, q.target_});
    q.targets_.push_back(q.target_);
  }

  if(q.source_ == 0) {
    for(raptor_edge edge : q.raptor_edges_start_) {
      q.sources_.push_back(edge.to_);
    }
  }
  else {
    q.raptor_edges_start_.push_back(raptor_edge{0, q.source_, q.source_});
    q.sources_.push_back(q.source_);
  }

  return q;
}

}  // namespace motis::mcraptor
