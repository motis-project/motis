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
                                    bestRouteLabels(std::vector<Bag>()),
                                    bestTransferLabels(std::vector<Bag>()),
                                    stopsForTransfers(* new cpu_mark_store(q.tt_.stop_count())),
                                    stopsForRoutes(* new cpu_mark_store(q.tt_.stop_count())),
                                    round(0) {};

/*
  trip_count get_earliest_trip(raptor_timetable const& tt,
                               raptor_route const& route,
                               time const* prev_arrivals,
                               stop_times_index r_stop_offset);
*/

  void init_arrivals();
/*
  void update_route(raptor_timetable const& tt, route_id r_id,
                    time const* prev_arrivals, time* current_round,
                    earliest_arrivals& ea, cpu_mark_store& station_marks);

  void update_footpaths(raptor_timetable const& tt, time* current_round,
                        earliest_arrivals const& ea,
                        cpu_mark_store& station_marks);*/

  //void invoke_cpu_raptor(raptor_query const& query, raptor_statistics&);

  void invoke_cpu_raptor();

  Bag* currentRound();
  Bag* previousRound();
  void startNewRound();

  void arrival_by_route(stop_id stop, Label& newLabel);
  void arrival_by_transfer(stop_id stop, Label& label);
  void relax_transfers();
  void collect_routes_serving_updated_stops();
  void scan_routes();

  inline std::vector<std::pair<route_id, stop_id>> getRoutesTimesForStop(stop_id stop);


public:

  raptor_query const& query;
  Rounds& result;
  int round;

  std::vector<Bag> routeLabels;
  std::vector<Bag> transferLabels;
  std::map<route_id, stop_id> routesServingUpdatedStops;
  cpu_mark_store& stopsForTransfers;
  cpu_mark_store& stopsForRoutes;

};


}  // namespace motis::mcraptor