#pragma once

#include <algorithm>

#include "motis/core/common/timing.h"
#include "motis/core/schedule/interval.h"
#include "motis/core/schedule/schedule.h"

#include "motis/csa/collect_start_times.h"
#include "motis/csa/csa_query.h"
#include "motis/csa/csa_statistics.h"
#include "motis/csa/csa_timetable.h"
#include "motis/csa/pareto_set.h"
#include "motis/csa/response.h"

namespace motis::csa {

template <typename SearchStrategy>
struct pretrip : public SearchStrategy {
  pretrip(schedule const& sched, csa_timetable const& tt, csa_query const& q,
          csa_statistics& stats)
      : SearchStrategy{sched, tt, q, stats},
        schedule_begin_{SCHEDULE_OFFSET_MINUTES},
        schedule_end_{static_cast<motis::time>(
            (sched.schedule_end_ - sched.schedule_begin_) / 60)} {}

  response search() {
    MOTIS_START_TIMING(total_timing);

    static_cast<SearchStrategy*>(this)->search_in_interval(
        results_, search_interval_, true);
    while (!min_connection_count_reached() && !max_interval_reached()) {
      auto const extended_search_interval =
          interval{query().extend_interval_earlier_
                       ? map_to_interval(search_interval_.begin_ - 60)
                       : search_interval_.begin_,
                   query().extend_interval_later_  //
                       ? map_to_interval(search_interval_.end_ + 60)
                       : search_interval_.end_};

      if (extended_search_interval.begin_ != search_interval_.begin_) {
        static_cast<SearchStrategy*>(this)->search_in_interval(
            results_,
            interval{extended_search_interval.begin_,
                     map_to_interval(search_interval_.begin_ - 1)},
            false);
      }

      if (extended_search_interval.end_ != search_interval_.end_) {
        static_cast<SearchStrategy*>(this)->search_in_interval(
            results_,
            interval{map_to_interval(search_interval_.end_ + 1),
                     extended_search_interval.end_},
            true);
      }

      search_interval_ = extended_search_interval;
    }

    utl::erase_if(results_.set_,
                  [&](csa_journey const& j) { return !in_interval(j); });

    MOTIS_STOP_TIMING(total_timing);
    stats().total_duration_ = MOTIS_TIMING_MS(total_timing);

    return {stats(), std::move(results_.set_), search_interval_};
  }

private:
  static bool dominates(csa_journey const& a, csa_journey const& b) {
    return a.journey_begin() >= b.journey_begin() &&
           a.journey_end() <= b.journey_end() && a.transfers_ <= b.transfers_;
  }

  bool in_interval(csa_journey const& j) const {
    return j.journey_begin() >= search_interval_.begin_ &&
           j.journey_begin() <= search_interval_.end_;
  }

  bool min_connection_count_reached() const {
    return std::count_if(begin(results_), end(results_),
                         [&](csa_journey const& j) {
                           return in_interval(j);
                         }) >= query().min_connection_count_;
  }

  bool max_interval_reached() const {
    return (!query().extend_interval_earlier_ ||
            search_interval_.begin_ == schedule_begin_) &&
           (!query().extend_interval_later_ ||  //
            search_interval_.end_ == schedule_end_);
  }

  motis::time map_to_interval(time const t) const {
    return std::min(schedule_end_, std::max(schedule_begin_, t));
  }

  csa_query const& query() const { return SearchStrategy::q_; }
  csa_statistics& stats() const { return SearchStrategy::stats_; }

  motis::time schedule_begin_, schedule_end_;
  interval search_interval_{
      map_to_interval(SearchStrategy::q_.search_interval_.begin_),
      map_to_interval(SearchStrategy::q_.search_interval_.end_)};
  pareto_set<csa_journey, decltype(&dominates)> results_{
      make_pareto_set<csa_journey>(&dominates)};
};

template <typename CSASearch>
struct pretrip_iterated_ontrip_search {
  pretrip_iterated_ontrip_search(schedule const& sched, csa_timetable const& tt,
                                 csa_query const& q, csa_statistics& stats)
      : sched_{sched}, tt_{tt}, q_{q}, stats_{stats} {}

  template <typename Results>
  void search_in_interval(Results& results, interval const& search_interval,
                          bool const ontrip_at_interval_end) {
    auto const start_times =
        collect_start_times(tt_, q_, search_interval, ontrip_at_interval_end);
    for (auto const& start_time : start_times) {
      CSASearch csa{tt_, start_time, stats_};
      for (auto const& start_idx : q_.meta_starts_) {
        csa.add_start(tt_.stations_.at(start_idx), 0);
      }

      MOTIS_START_TIMING(search_timing);
      csa.search();
      MOTIS_STOP_TIMING(search_timing);

      MOTIS_START_TIMING(reconstruction_timing);
      collect_results(csa, results);
      MOTIS_STOP_TIMING(reconstruction_timing);

      stats_.search_duration_ += MOTIS_TIMING_MS(search_timing);
      stats_.reconstruction_duration_ += MOTIS_TIMING_MS(reconstruction_timing);
    }
  }

  template <typename Results>
  void collect_results(CSASearch& csa, Results& results) {
    for (auto const& dest_idx : q_.meta_dests_) {
      for (csa_journey& j : csa.get_results(tt_.stations_.at(dest_idx),
                                            q_.include_equivalent_)) {
        if (j.duration() <= MAX_TRAVEL_TIME) {
          results.push_back(j);
        }
      }
    }
  }

  schedule const& sched_;
  csa_timetable const& tt_;
  csa_query const& q_;
  csa_statistics& stats_;
};

template <typename CSAProfileSearch, typename CSAOnTripSearch>
struct pretrip_profile_search {
  pretrip_profile_search(schedule const& sched, csa_timetable const& tt,
                         csa_query const& q, csa_statistics& stats)
      : sched_{sched}, tt_{tt}, q_{q}, stats_{stats} {}

  template <typename Results>
  void search_in_interval(Results& results, interval const& search_interval,
                          bool const ontrip_at_interval_end) {
    CSAProfileSearch profile_csa{tt_, search_interval, stats_};
    run_search(profile_csa, results);

    if (ontrip_at_interval_end) {
      CSAOnTripSearch ontrip_csa{
          tt_, static_cast<time>(search_interval.end_ + 1), stats_};
      run_search(ontrip_csa, results);
    }
  }

  template <typename CSASearch, typename Results>
  void run_search(CSASearch& csa, Results& results) {
    for (auto const& start_idx : q_.meta_starts_) {
      csa.add_start(tt_.stations_.at(start_idx), 0);
    }

    MOTIS_START_TIMING(search_timing);
    csa.search();
    MOTIS_STOP_TIMING(search_timing);

    MOTIS_START_TIMING(reconstruction_timing);
    collect_results(csa, results);
    MOTIS_STOP_TIMING(reconstruction_timing);

    stats_.search_duration_ += MOTIS_TIMING_MS(search_timing);
    stats_.reconstruction_duration_ += MOTIS_TIMING_MS(reconstruction_timing);
  }

  template <typename CSASearch, typename Results>
  void collect_results(CSASearch& csa, Results& results) {
    for (auto const& dest_idx : q_.meta_dests_) {
      for (csa_journey& j : csa.get_results(tt_.stations_.at(dest_idx),
                                            q_.include_equivalent_)) {
        if (j.duration() <= MAX_TRAVEL_TIME) {
          results.push_back(j);
        }
      }
    }
  }

  schedule const& sched_;
  csa_timetable const& tt_;
  csa_query const& q_;
  csa_statistics& stats_;
};

}  // namespace motis::csa
