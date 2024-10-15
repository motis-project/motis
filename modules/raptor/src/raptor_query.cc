#include "motis/raptor/raptor_query.h"

#include "motis/core/access/error.h"

#include "utl/verify.h"

namespace motis::raptor {

using namespace motis::routing;

stop_id checked_eva_to_raptor_id(raptor_meta_info const& meta_info,
                                 std::string const& eva) {
  try {
    return meta_info.eva_to_raptor_id_.at(eva);
  } catch (std::out_of_range const&) {
    throw std::system_error{access::error::station_not_found};
  }
}

base_query get_base_query(RoutingRequest const* routing_request,
                          schedule const& sched,
                          raptor_meta_info const& meta_info) {
  utl::verify(routing_request->search_dir() == SearchDir_Forward,
              "RAPTOR currently only supports departure queries");
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

  q.source_ = checked_eva_to_raptor_id(meta_info, start_eva);
  q.target_ = checked_eva_to_raptor_id(meta_info, target_eva);

  q.forward_ = (routing_request->search_dir() == SearchDir::SearchDir_Forward);
  q.ontrip_ = routing_request->start_type() != Start::Start_PretripStart;

  // Don't use start meta stations in the ontrip case, because:
  // do as the routing module does
  q.use_start_metas_ =
      routing_request->use_start_metas() &&
      routing_request->start_type() != Start_OntripStationStart;
  q.use_dest_metas_ = routing_request->use_dest_metas();

  q.use_start_footpaths_ = routing_request->use_start_footpaths();

  q.criteria_config_ =
      get_criteria_config_from_search_type(routing_request->search_type());

  return q;
}

}  // namespace motis::raptor
