#pragma once

#include "motis/mcraptor/cpu/mark_store.h"

#include "motis/mcraptor/raptor_query.h"
#include "motis/mcraptor/raptor_result.h"
#include "motis/mcraptor/raptor_statistics.h"
#include "motis/mcraptor/raptor_timetable.h"

namespace motis::mcraptor {

struct McRaptor {

  McRaptor(raptor_query const& q) : query(q),
                                    result(q.result()),
                                    routeLabels(q.tt_.stop_count()),
                                    transferLabels(q.tt_.stop_count()),
                                    stopsForTransfers(* new cpu_mark_store(q.tt_.stop_count())),
                                    stopsForRoutes(* new cpu_mark_store(q.tt_.stop_count())),
                                    round(-1) {};

  void invoke_cpu_raptor();

  Bag* currentRound();
  Bag* previousRound();
  void startNewRound();

  void init_arrivals();
  void arrival_by_route(stop_id stop, Label& newLabel);
  void arrival_by_transfer(stop_id stop, Label& label);
  void relax_transfers();
  void collect_routes_serving_updated_stops();
  void scan_routes();

  inline std::vector<std::pair<route_id, route_stops_index>> getRoutesTimesForStop(stop_id stop);


public:

  raptor_query const& query;
  Rounds& result;
  int round;

  std::vector<Bag> routeLabels;
  std::vector<Bag> transferLabels;
  std::map<route_id, route_stops_index> routesServingUpdatedStops;
  cpu_mark_store& stopsForTransfers;
  cpu_mark_store& stopsForRoutes;

};


}  // namespace motis::mcraptor