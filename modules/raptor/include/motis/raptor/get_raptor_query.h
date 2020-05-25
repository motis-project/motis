#pragma once

#include "motis/raptor/raptor_query.h"

#include "motis/core/schedule/schedule.h"
#include "motis/core/access/error.h"
#include "motis/module/message.h"

namespace motis::raptor {

using namespace motis::routing;

base_query get_base_query(RoutingRequest const* routing_request,
                          schedule const& sched,
                          raptor_schedule const& raptor_sched) {
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

    default: {
      throw std::system_error(access::error::not_implemented);
    } break;
  }

  q.source_ = raptor_sched.eva_to_raptor_id_.at(start_eva);
  q.target_ = raptor_sched.eva_to_raptor_id_.at(target_eva);

  q.forward_ = (routing_request->search_dir() == SearchDir::SearchDir_Forward);

  if (!q.forward_) {
    q.source_time_begin_ = -q.source_time_begin_;
    q.source_time_end_ = -q.source_time_end_;
  }

  q.use_dest_metas_ = routing_request->use_dest_metas();

  return q;
}

}  // namespace motis::raptor