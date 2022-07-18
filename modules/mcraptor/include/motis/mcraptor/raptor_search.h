#include <fstream>

#include "motis/core/common/timing.h"
#include "motis/core/schedule/time.h"
#include "motis/core/journey/journey.h"

#include "motis/mcraptor/cpu/cpu_raptor.h"
#include "motis/mcraptor/raptor_timetable.h"
#include "motis/mcraptor/raptor_util.h"
#include "motis/mcraptor/reconstructor.h"

#if defined(MOTIS_CUDA)
#include "motis/mcraptor/gpu/gpu_raptor.cuh"
#endif

namespace motis::mcraptor {

inline auto get_departure_range(time const begin, time const end,
                                std::vector<time> const& departure_events) {
  std::ptrdiff_t const lower =
      std::distance(std::cbegin(departure_events),
                    std::lower_bound(std::cbegin(departure_events),
                                     std::cend(departure_events), begin)) -
      1;
  std::ptrdiff_t const upper =
      std::distance(std::cbegin(departure_events),
                    std::upper_bound(std::cbegin(departure_events),
                                     std::cend(departure_events), end)) -
      1;

  return std::pair(lower, upper);
}

template <typename RaptorFun, typename Query>
inline std::vector<journey> raptor_gen(Query& q, raptor_statistics& stats,
                                       schedule const& sched,
                                       raptor_meta_info const& raptor_sched,
                                       raptor_timetable const& timetable,
                                       RaptorFun const& raptor_search) {
  reconstructor reconstructor(sched, raptor_sched, timetable);

  if (q.ontrip_) {
    stats.raptor_queries_ = 1;

    MOTIS_START_TIMING(raptor_time);
    raptor_search(q);
    stats.raptor_time_ = MOTIS_GET_TIMING_MS(raptor_time);

    MOTIS_START_TIMING(rec_timing);
    reconstructor.add(q);
    stats.rec_time_ = MOTIS_GET_TIMING_US(rec_timing);

    return reconstructor.get_journeys();
  }

  // Get departure range before we do the +1 query
  auto const& dep_events = q.use_start_metas_
                               ? raptor_sched.departure_events_with_metas_
                               : raptor_sched.departure_events_;
  auto const [lower, upper] = get_departure_range(
      q.source_time_begin_, q.source_time_end_, dep_events[q.source_]);

  stats.raptor_queries_ += 1;
  q.source_time_begin_ = q.source_time_end_ + 1;
  MOTIS_START_TIMING(plus_one_time);
  raptor_search(q);
  stats.raptor_time_ += MOTIS_GET_TIMING_US(plus_one_time);

  MOTIS_START_TIMING(plus_one_rec_time);
  reconstructor.add(q);
  stats.rec_time_ += MOTIS_GET_TIMING_US(plus_one_rec_time);

//  for (auto dep_idx = upper; dep_idx != lower; --dep_idx) {
//    stats.raptor_queries_ += 1;
//    q.source_time_begin_ = dep_events[q.source_][dep_idx];
//
//    MOTIS_START_TIMING(raptor_time);
//    raptor_search(q);
//    stats.raptor_time_ += MOTIS_GET_TIMING_US(raptor_time);
//
//    MOTIS_START_TIMING(rec_timing);
//    reconstructor.add(q);
//    stats.rec_time_ += MOTIS_GET_TIMING_US(rec_timing);
//  }

  //return reconstructor.get_journeys(q.source_time_end_);
  return reconstructor.get_journeys();
}

inline std::vector<journey> cpu_raptor(raptor_query& q,
                                       raptor_statistics& stats,
                                       schedule const& sched,
                                       raptor_meta_info const& raptor_sched,
                                       raptor_timetable const& tt) {
  return raptor_gen(q, stats, sched, raptor_sched, tt, [&](raptor_query& q) {
    McRaptor mcRaptor = McRaptor(q);
    return mcRaptor.invoke_cpu_raptor();
  });
}

#if defined(MOTIS_CUDA)
inline std::vector<journey> gpu_raptor(d_query& dq, raptor_statistics& stats,
                                       schedule const& sched,
                                       raptor_meta_info const& raptor_sched,
                                       raptor_timetable const& tt) {
  return raptor_gen(dq, stats, sched, raptor_sched, tt,
                    [&](d_query& dq) { return invoke_gpu_raptor(dq); });
}
#endif

}  // namespace motis::mcraptor
