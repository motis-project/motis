#include <fstream>

#include "motis/core/common/timing.h"
#include "motis/core/schedule/edges.h"
#include "motis/core/schedule/time.h"
#include "motis/core/journey/journey.h"
#include "motis/protocol/RoutingRequest_generated.h"

#include "motis/raptor/cpu/cpu_raptor.h"
#include "motis/raptor/cpu/mc_cpu_raptor.h"
#include "motis/raptor/criteria/configs.h"
#include "motis/raptor/implementation_type.h"
#include "motis/raptor/raptor_timetable.h"
#include "motis/raptor/raptor_util.h"
#include "motis/raptor/reconstructor.h"

#if defined(MOTIS_CUDA)
#include "motis/raptor/gpu/gpu_raptor.cuh"
#endif

namespace motis::raptor {

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

template <typename CriteriaConfig, typename RaptorFun, typename Query>
inline std::vector<journey> raptor_gen(Query& q, raptor_statistics& stats,
                                       schedule const& sched,
                                       raptor_meta_info const& raptor_sched,
                                       raptor_timetable const& timetable,
                                       RaptorFun const& raptor_search) {
  reconstructor<CriteriaConfig> reconstructor(sched, raptor_sched, timetable);

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

  for (auto dep_idx = upper; dep_idx != lower; --dep_idx) {
    stats.raptor_queries_ += 1;
    q.source_time_begin_ = dep_events[q.source_][dep_idx];

    MOTIS_START_TIMING(raptor_time);
    raptor_search(q);
    stats.raptor_time_ += MOTIS_GET_TIMING_US(raptor_time);

    MOTIS_START_TIMING(rec_timing);
    reconstructor.add(q);
    stats.rec_time_ += MOTIS_GET_TIMING_US(rec_timing);
  }

  return reconstructor.get_journeys(q.source_time_end_);
}

template <implementation_type Impl, typename Query>
inline std::vector<journey> search_dispatch(
    Query& q, raptor_statistics& stats, schedule const& sched,
    raptor_meta_info const& meta_info, raptor_timetable const& tt,
    routing::SearchType const search_type) {
  throw std::system_error{access::error::not_implemented};
}

template <>
inline std::vector<journey>
search_dispatch<implementation_type::CPU, raptor_query>(
    raptor_query& q, raptor_statistics& stats, schedule const& sched,
    raptor_meta_info const& meta_info, raptor_timetable const& tt,
    routing::SearchType const search_type) {
  switch (search_type) {
    case routing::SearchType::SearchType_Default:
      return raptor_gen<Default>(
          q, stats, sched, meta_info, tt,
          [&](raptor_query& q) { return invoke_cpu_raptor(q, stats); });

    case routing::SearchType::SearchType_MaxOccupancy:
      q.result_.reset();
      q.result_ = std::make_unique<raptor_result>(tt.stop_count() *
                                                  MaxOccupancy::trait_size());
      return raptor_gen<MaxOccupancy>(
          q, stats, sched, meta_info, tt, [&](raptor_query& q) {
            return invoke_mc_cpu_raptor<MaxOccupancy>(q, stats);
          });

    default: throw std::system_error{access::error::not_implemented};
  }
}

#if defined(MOTIS_CUDA)
template <>
inline std::vector<journey> search_dispatch<implementation_type::GPU, d_query>(
    d_query& q, raptor_statistics& stats, schedule const& sched,
    raptor_meta_info const& meta_info, raptor_timetable const& tt,
    routing::SearchType const search_type) {
  utl::verify_ex(search_type == routing::SearchType_Default,
                 access ::error::not_implemented);
  return raptor_gen<Default>(q, stats, sched, meta_info, tt,
                             [&](d_query& q) { return invoke_gpu_raptor(q); });
}
#endif

// inline std::vector<journey> cpu_raptor(raptor_query& q,
//                                        raptor_statistics& stats,
//                                        schedule const& sched,
//                                        raptor_meta_info const& raptor_sched,
//                                        raptor_timetable const& tt) {
//   return raptor_gen(q, stats, sched, raptor_sched, tt, [&](raptor_query& q) {
//     return invoke_cpu_raptor(q, stats);
//   });
// }
//
//#if defined(MOTIS_CUDA)
// inline std::vector<journey> gpu_raptor(d_query& dq, raptor_statistics& stats,
//                                        schedule const& sched,
//                                        raptor_meta_info const& raptor_sched,
//                                        raptor_timetable const& tt) {
//   return raptor_gen(dq, stats, sched, raptor_sched, tt,
//                     [&](d_query& dq) { return invoke_gpu_raptor(dq); });
// }
//#endif

}  // namespace motis::raptor
