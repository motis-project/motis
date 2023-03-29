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

template <class L, class MCRaptor>
inline std::vector<journey> raptor_gen(raptor_query<L>& q, raptor_statistics& stats,
                                       schedule const& sched,
                                       raptor_meta_info const& raptor_sched,
                                       raptor_timetable const& timetable,
                                       MCRaptor raptor) {
  reconstructor reconstructor(sched, raptor_sched, timetable);
//  std::cout << "SCHED BEGIN: " << sched.schedule_begin_ << std::endl;
//  std::cout << "NEED TIME: " << unix_to_motistime(sched.schedule_begin_, 1655021460) << std::endl;
//  std::cout << "INTERVAL_BEGIN: " << motis_to_unixtime(sched.schedule_begin_, q.source_time_begin_) << std::endl;
//  std::cout << "INTERVAL_END: " << motis_to_unixtime(sched.schedule_begin_, q.source_time_end_) << std::endl;
//
//  std::cout << "SOURCE: " << q.source_ << "; " << q.meta_info_.raptor_id_to_eva_[q.source_] << std::endl;
//  std::cout << "TARGET: " << q.target_ << "; " << q.meta_info_.raptor_id_to_eva_[q.target_] << std::endl;
//
//  std::cout << "Current station raptor id: " << q.meta_info_.eva_to_raptor_id_.at("8502350") << std::endl;
//  std::cout << "NEED station raptor id: " << q.meta_info_.eva_to_raptor_id_.at("8572745") << std::endl;

  // Get departure range before we do the +1 query
  std::vector<std::vector<time>> const& dep_events = q.use_start_metas_
                               ? raptor_sched.departure_events_with_metas_
                               : raptor_sched.departure_events_;

  std::vector<std::vector<time>> const& arr_events = q.use_start_metas_
                               ? raptor_sched.arrival_events_with_metas_
                               : raptor_sched.arrival_events_;
  //TODO MERGE WITH INTERMODAL
  std::vector<time> const& events = q.forward_ ? dep_events[q.source_] : arr_events[q.target_];
  auto const [lower, upper] = get_departure_range(
      q.source_time_begin_, q.source_time_end_, events);
  MOTIS_START_TIMING(raptor_time);

  //TODO FIX THIS - make virtual method in cpu raptor and implement it for forward and backward
  if(q.forward_) {
    for (raptor_edge s: q.raptor_edges_start_) {
      /*std::cout << "New Start Edge" << std::endl;
      std::cout << "\tAnalyzing start edge: " << q.meta_info_.raptor_id_to_eva_.at(s.to_) << " with duration " << s.duration_ << std::endl;*/
      auto const [lower, upper] = get_departure_range(
          q.source_time_begin_ + s.duration_, q.source_time_end_ + s.duration_, dep_events[s.to_]);

      stats.raptor_queries_ += 1;
      raptor.reset();
      raptor.set_current_start_edge(s);
      raptor.set_query_source_time(q.source_time_end_ + s.duration_ + 1);
      raptor.invoke_cpu_raptor();


      for (auto dep_idx = upper; dep_idx != lower; --dep_idx) {
        raptor.reset();
        stats.raptor_queries_ += 1;
        time new_query_time = dep_events[s.to_][dep_idx];

        raptor.set_query_source_time(new_query_time);
        raptor.invoke_cpu_raptor();
      }

      raptor.reset();
      raptor.set_query_source_time(q.source_time_begin_ + s.duration_);
      raptor.invoke_cpu_raptor();
    }
  }
  else {
    stats.raptor_queries_ += 1;
    raptor.set_query_source_time(q.source_time_begin_ - 1);
    raptor.invoke_cpu_raptor();


    for (auto dep_idx = lower; dep_idx != upper; ++dep_idx) {
      raptor.reset();
      stats.raptor_queries_ += 1;
      time new_query_time = arr_events[q.target_][dep_idx];

//      std::cout << "TIME BEGIN: " << new_query_time << "; " << motis_to_unixtime(sched.schedule_begin_, new_query_time) << std::endl;

      raptor.set_query_source_time(new_query_time);
      raptor.invoke_cpu_raptor();
    }

    raptor.reset();
    raptor.set_query_source_time(q.source_time_end_);
    raptor.invoke_cpu_raptor();

    raptor.init_parents();
  }

  stats.raptor_time_ += MOTIS_GET_TIMING_US(raptor_time);

  MOTIS_START_TIMING(plus_one_rec_time);
  reconstructor.add(q);
  stats.rec_time_ += MOTIS_GET_TIMING_US(plus_one_rec_time);
  return reconstructor.get_journeys(q.source_time_end_);
}

inline std::vector<journey> cpu_raptor(base_query& bq,
                                       raptor_statistics& stats,
                                       schedule const& sched,
                                       raptor_meta_info const& raptor_sched,
                                       raptor_timetable const& tt) {
  if(bq.forward_) {
    raptor_query<label_departure> q =
        raptor_query<label_departure>{bq, raptor_sched, tt};
    return raptor_gen<label_departure, mc_raptor_departure>(q, stats, sched, raptor_sched, tt, mc_raptor_departure(q));
  }
  else {
    raptor_query<label_backward> q =
        raptor_query<label_backward>{bq, raptor_sched, tt};
    return raptor_gen<label_backward, mc_raptor_backward>(q, stats, sched, raptor_sched, tt, mc_raptor_backward(q));
  }



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
