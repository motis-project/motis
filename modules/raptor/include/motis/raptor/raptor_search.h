#include <fstream>

#include "motis/core/common/timing.h"
#include "motis/core/schedule/time.h"
#include "motis/core/journey/journey.h"

#include "motis/raptor/cpu_raptor.h"
#include "motis/raptor/raptor_timetable.h"
#include "motis/raptor/raptor_util.h"
#include "motis/raptor/reconstructor.h"

namespace motis::raptor {

inline auto get_departure_range(time const begin, time const end,
                                std::vector<time> const& departure_events) {

  auto const lower = std::lower_bound(std::cbegin(departure_events),
                                      std::cend(departure_events), begin) -
                     1;
  auto const upper = std::upper_bound(std::cbegin(departure_events),
                                      std::cend(departure_events), end) -
                     1;

  return std::pair(lower, upper);
}

template <typename RaptorFun, typename Query>
inline std::vector<journey> bw_search(
    Query& q, raptor_statistics&, schedule const& sched,
    raptor_schedule const& raptor_sched, raptor_timetable const&,
    raptor_timetable const& backward_timetable,
    RaptorFun const& raptor_search) {
  std::cout << "BW SEARCH\n";

  std::ofstream id_to_eva("id_to_eva.txt");
  for (auto const& [id, eva] : raptor_sched.eva_to_raptor_id_) {
    id_to_eva << id << " " << eva << '\n';
  }
  id_to_eva.close();

  reconstructor reconstructor(q, sched, raptor_sched, backward_timetable);

  print_station_arrivals(q.target_, *q.result_);
  std::cout << "Query Source: " << q.source_ << '\n';
  std::cout << "Query Target: " << q.target_ << '\n';
  raptor_search(q);

  print_station_arrivals(q.source_, *q.result_);
  print_station_arrivals(q.target_, *q.result_);

  print_station_arrivals(255818, *q.result_);
  print_station_arrivals(255870, *q.result_);
  reconstructor.add(q.source_time_begin_, *q.result_, q.forward_);

  // print_station_arrivals(q.target_, *q.result_);

  return reconstructor.get_journeys();
}

template <typename RaptorFun, typename Query>
inline std::vector<journey> raptor(Query& q, raptor_statistics& stats,
                                   schedule const& sched,
                                   raptor_schedule const& raptor_sched,
                                   raptor_timetable const& timetable,
                                   raptor_timetable const& backward_timetable,
                                   RaptorFun const& raptor_search) {
  if (!q.forward_) {
    return bw_search(q, stats, sched, raptor_sched, timetable,
                     backward_timetable, raptor_search);
  }

  reconstructor reconstructor(q, sched, raptor_sched, timetable);

  // We have a ontrip query, just a single raptor query is needed
  if (q.source_time_begin_ == q.source_time_end_) {
    stats.raptor_queries_ = 1;

    MOTIS_START_TIMING(raptor_time);
    raptor_search(q);
    stats.raptor_time_ = MOTIS_GET_TIMING_MS(raptor_time);

    MOTIS_START_TIMING(rec_timing);
    reconstructor.add(q.source_time_begin_, *q.result_, q.forward_);
    stats.rec_time_ = MOTIS_GET_TIMING_US(rec_timing);

    return reconstructor.get_journeys();
  }

  // Get departure range before we do the +1 query
  auto const& [lower, upper] =
      get_departure_range(q.source_time_begin_, q.source_time_end_,
                          raptor_sched.departure_events_[q.source_]);

  stats.raptor_queries_ += 1;
  q.source_time_begin_ = q.source_time_end_ + 1;
  MOTIS_START_TIMING(plus_one_time);
  raptor_search(q);
  stats.raptor_time_ += MOTIS_GET_TIMING_US(plus_one_time);

  MOTIS_START_TIMING(plus_one_rec_time);
  reconstructor.add(q.source_time_begin_, *q.result_, q.forward_);
  stats.rec_time_ += MOTIS_GET_TIMING_US(plus_one_rec_time);

  for (auto dep_it = upper; dep_it != lower; --dep_it) {
    stats.raptor_queries_ += 1;
    q.source_time_begin_ = *dep_it;

    MOTIS_START_TIMING(raptor_time);
    raptor_search(q);
    stats.raptor_time_ += MOTIS_GET_TIMING_US(raptor_time);

    MOTIS_START_TIMING(rec_timing);
    reconstructor.add(q.source_time_begin_, *q.result_, q.forward_);
    stats.rec_time_ += MOTIS_GET_TIMING_US(rec_timing);
  }

  return reconstructor.get_journeys(q.source_time_end_);
}

inline std::vector<journey> cpu_raptor(raptor_query& q,
                                       raptor_statistics& stats,
                                       schedule const& sched,
                                       raptor_schedule const& raptor_sched,
                                       raptor_timetable const& tt,
                                       raptor_timetable const& btt) {
  return raptor(q, stats, sched, raptor_sched, tt, btt,
                [&](raptor_query& q) { return invoke_cpu_raptor(q, stats); });
}

}  // namespace motis::raptor
