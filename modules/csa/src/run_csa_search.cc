#include "motis/csa/run_csa_search.h"

#include "utl/verify.h"

#include "motis/core/common/timing.h"

#ifdef MOTIS_AVX
#include "motis/csa/cpu/csa_search_default_cpu_sse.h"
#endif
#ifdef MOTIS_CUDA
#include "motis/csa/gpu/gpu_search.h"
#endif
#include "motis/csa/cpu/csa_profile_search_default_cpu.h"
#include "motis/csa/cpu/csa_search_default_cpu.h"
#include "motis/csa/error.h"
#include "motis/csa/pareto_set.h"
#include "motis/csa/pretrip.h"

using namespace motis::routing;

namespace motis::csa {

inline auto make_ontrip_pareto_set(bool const include_equivalent) {
  return make_pareto_set<csa_journey>(
      [include_equivalent](csa_journey const& a, csa_journey const& b) {
        if (include_equivalent) {
          // price is currently not used
          return (a.journey_end() < b.journey_end() &&
                  a.transfers_ <= b.transfers_) ||
                 (a.journey_end() <= b.journey_end() &&
                  a.transfers_ < b.transfers_);
        } else {
          return (a.journey_end() <= b.journey_end() &&
                  a.transfers_ <= b.transfers_ && a.price_ <= b.price_);
        }
      });
}

template <typename CSASearch>
response run_search(schedule const& sched, csa_timetable const& tt,
                    csa_query const& q) {
  csa_statistics stats;

  if (q.is_ontrip()) {
    MOTIS_START_TIMING(total_timing);
    CSASearch csa(tt, q.search_interval_.begin_, stats);
    for (auto const& start_idx : q.meta_starts_) {
      csa.add_start(tt.stations_.at(start_idx), 0);
    }

    MOTIS_START_TIMING(search_timing);
    csa.search();
    MOTIS_STOP_TIMING(search_timing);

    MOTIS_START_TIMING(reconstruction_timing);
    auto results = make_ontrip_pareto_set(q.include_equivalent_);
    for (auto const& dest_idx : q.meta_dests_) {
      for (auto j :
           csa.get_results(tt.stations_.at(dest_idx), q.include_equivalent_)) {
        results.push_back(j);
      }
    }
    MOTIS_STOP_TIMING(reconstruction_timing);
    MOTIS_STOP_TIMING(total_timing);

    stats.search_duration_ = MOTIS_TIMING_MS(search_timing);
    stats.reconstruction_duration_ = MOTIS_TIMING_MS(reconstruction_timing);
    stats.total_duration_ = MOTIS_TIMING_MS(total_timing);

    return {stats, std::move(results.set_), q.search_interval_};
  } else {
    return pretrip<pretrip_iterated_ontrip_search<CSASearch>>(sched, tt, q,
                                                              stats)
        .search();
  }
}

template <typename CSAProfileSearch, typename CSAOnTripSearch>
response run_profile_search(schedule const& sched, csa_timetable const& tt,
                            csa_query const& q) {
  if (q.is_ontrip()) {
    return run_search<CSAOnTripSearch>(sched, tt, q);
  } else {
    csa_statistics stats;
    return pretrip<pretrip_profile_search<CSAProfileSearch, CSAOnTripSearch>>(
               sched, tt, q, stats)
        .search();
  }
}

template <search_dir Dir>
response dispatch_search_type(schedule const& sched, csa_timetable const& tt,
                              csa_query const& q, SearchType const search_type,
                              implementation_type const impl_type,
                              bool use_profile_search) {
  switch (impl_type) {
    case implementation_type::CPU:
      switch (search_type) {
        case SearchType_Default:
        case SearchType_Accessibility:
          return use_profile_search
                     ? run_profile_search<cpu::csa_profile_search<Dir>,
                                          cpu::csa_search<Dir>>(sched, tt, q)
                     : run_search<cpu::csa_search<Dir>>(sched, tt, q);
        default: throw std::system_error(error::search_type_not_supported);
      }

#ifdef MOTIS_AVX
    case implementation_type::CPU_SSE:
      switch (search_type) {
        case SearchType_Default:
        case SearchType_Accessibility:
          return run_search<cpu::sse::csa_search<Dir>>(sched, tt, q);
        default: throw std::system_error(error::search_type_not_supported);
      }
#endif

#ifdef MOTIS_CUDA
    case implementation_type::GPU:
      if constexpr (Dir == search_dir::BWD) {
        throw std::system_error(error::search_type_not_supported);
      } else {
        switch (search_type) {
          case SearchType_Default:
          case SearchType_Accessibility: {
            utl::verify_ex(
                !q.include_equivalent_,
                std::system_error{error::include_equivalent_not_supported});
            csa_statistics stats;
            if (q.is_ontrip()) {
              gpu_search s{sched, tt, q, stats};
              auto results = make_ontrip_pareto_set(q.include_equivalent_);
              s.search(results, {q.search_interval_.begin_});
              return {stats, std::move(results.set_), q.search_interval_};
            } else {
              return pretrip<gpu_search>(sched, tt, q, stats).search();
            }
          }
          default: throw std::system_error(error::search_type_not_supported);
        }
      }
#endif

    default: throw std::system_error(error::search_type_not_supported);
  }
}

response run_csa_search(schedule const& sched, csa_timetable const& tt,
                        csa_query const& q, SearchType const search_type,
                        implementation_type const impl_type,
                        bool use_profile_search) {
  if ((tt.fwd_connections_.empty() && q.dir_ == search_dir::FWD) ||
      (tt.bwd_connections_.empty() && q.dir_ == search_dir::BWD)) {
    response r;
    r.searched_interval_ = q.search_interval_;
    return r;
  }

  return q.dir_ == search_dir::FWD
             ? dispatch_search_type<search_dir::FWD>(
                   sched, tt, q, search_type, impl_type, use_profile_search)
             : dispatch_search_type<search_dir::BWD>(
                   sched, tt, q, search_type, impl_type, use_profile_search);
}

}  // namespace motis::csa
