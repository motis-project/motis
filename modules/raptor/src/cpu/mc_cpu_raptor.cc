#include "motis/raptor/cpu/mc_cpu_raptor.h"

#include "motis/raptor/criteria/configs.h"
#include "motis/raptor/cpu/mark_store.h"
#include "motis/raptor/print_raptor.h"
#include "motis/raptor/raptor_query.h"
#include "motis/raptor/raptor_result.h"
#include "motis/raptor/raptor_statistics.h"
#include "motis/raptor/raptor_timetable.h"

namespace motis::raptor {

template <typename CriteriaConfig>
trip_count get_earliest_trip(raptor_timetable const& tt,
                             raptor_route const& route,
                             time const* const prev_arrivals,
                             stop_times_index const r_stop_offset,
                             uint32_t trait_offset) {

  stop_id const stop_id =
      tt.route_stops_[route.index_to_route_stops_ + r_stop_offset];
  auto const stop_arr_idx =
      CriteriaConfig::get_arrival_idx(stop_id, trait_offset);

  // station was never visited, there can't be a earliest trip
  if (!valid(prev_arrivals[stop_arr_idx])) {
    return invalid<trip_count>;
  }

  // get first defined earliest trip for the stop in the route
  auto const first_trip_stop_idx = route.index_to_stop_times_ + r_stop_offset;
  auto const last_trip_stop_idx =
      first_trip_stop_idx + ((route.trip_count_ - 1) * route.stop_count_);

  trip_count current_trip = 0;
  for (auto stop_time_idx = first_trip_stop_idx;
       stop_time_idx <= last_trip_stop_idx;
       stop_time_idx += route.stop_count_) {

    auto const stop_time = tt.stop_times_[stop_time_idx];
    if (valid(stop_time.departure_) &&
        prev_arrivals[stop_arr_idx] <= stop_time.departure_) {
      return current_trip;
    }

    ++current_trip;
  }

  return invalid<trip_count>;
}

template <typename CriteriaConfig>
inline void init_arrivals(raptor_result& result, raptor_query const& q,
                          cpu_mark_store& station_marks) {

  auto const traits_size = CriteriaConfig::trait_size();
  auto propagate_across_traits = [traits_size](time* const arrivals,
                                               stop_id stop_id,
                                               motis::time arrival_val) {
    auto const last_arr_idx = (stop_id * traits_size) + traits_size;
    for (int arr_idx = (stop_id * traits_size); arr_idx < last_arr_idx;
         ++arr_idx) {
      arrivals[arr_idx] = std::min(arrival_val, arrivals[arr_idx]);
    }
  };

  propagate_across_traits(result[0], q.source_, q.source_time_begin_);
  station_marks.mark(q.source_);

  for (auto const& add_start : q.add_starts_) {
    motis::time const add_start_time = q.source_time_begin_ + add_start.offset_;
    propagate_across_traits(result[0], add_start.s_id_, add_start_time);
    station_marks.mark(add_start.s_id_);
  }
}

template <typename CriteriaConfig>
inline std::tuple<trip_id, typename CriteriaConfig::CriteriaData>
get_next_feasible_trip(raptor_timetable const& tt,
                       time const* const prev_arrivals, route_id r_id,
                       trip_id earliest_trip_id, uint32_t trait_offset,
                       stop_id arr_offset) {

  auto const& route = tt.routes_[r_id];
  stop_id new_dep_offset;

  typename CriteriaConfig::CriteriaData trip_data{};

  for (trip_id trip_id = earliest_trip_id + 1; trip_id < route.trip_count_;
       ++trip_id) {

    // aggregate the trait data for the current departure station
    new_dep_offset = arr_offset - 1;
    stop_times_index const arr_sti =
        route.index_to_stop_times_ + (trip_id * route.stop_count_) + arr_offset;
    CriteriaConfig::update_traits_aggregate(trip_data, tt, r_id, trip_id,
                                            arr_offset, arr_sti);

    do {
      auto const dep_sti = route.index_to_stop_times_ +
                           (trip_id * route.stop_count_) + new_dep_offset;
      auto const dep_s_id =
          tt.route_stops_[route.index_to_route_stops_ + new_dep_offset];
      auto const dep_arr_idx =
          CriteriaConfig::get_arrival_idx(dep_s_id, trait_offset);
      auto dep_stop_times = tt.stop_times_[dep_sti];

      if (CriteriaConfig::is_update_required(trip_data, trait_offset) &&
          valid(dep_stop_times.departure_) &&
          prev_arrivals[dep_arr_idx] <= dep_stop_times.departure_) {
        return std::make_tuple(trip_id, std::move(trip_data));
      }

      CriteriaConfig::update_traits_aggregate(trip_data, tt, r_id, trip_id,
                                              new_dep_offset, dep_sti);
      --new_dep_offset;

    } while (new_dep_offset >= 0 &&
             CriteriaConfig::is_update_required(trip_data, trait_offset));

    CriteriaConfig::reset_traits_aggregate(trip_data);
  }

  return std::make_tuple(invalid<raptor::trip_id>, std::move(trip_data));
}

template <typename CriteriaConfig>
void update_route_for_trait_offset(raptor_timetable const& tt,
                                   route_id const r_id,
                                   time const* const prev_arrivals,
                                   time* const current_round,
                                   earliest_arrivals& ea,
                                   cpu_mark_store& station_marks,
                                   uint32_t trait_offset, int target_arr_idx) {
  auto const& route = tt.routes_[r_id];

  typename CriteriaConfig::CriteriaData criteria_data{};

  trip_count earliest_trip_id = invalid<trip_count>;
  for (stop_id r_stop_offset = 0; r_stop_offset < route.stop_count_;
       ++r_stop_offset) {

    if (!valid(earliest_trip_id)) {
      earliest_trip_id = get_earliest_trip<CriteriaConfig>(
          tt, route, prev_arrivals, r_stop_offset, trait_offset);

      CriteriaConfig::reset_traits_aggregate(criteria_data);
      continue;
    }

    auto const stop_id =
        tt.route_stops_[route.index_to_route_stops_ + r_stop_offset];
    auto current_stop_time_idx = route.index_to_stop_times_ +
                                 (earliest_trip_id * route.stop_count_) +
                                 r_stop_offset;

    CriteriaConfig::update_traits_aggregate(criteria_data, tt, r_id,
                                            earliest_trip_id, r_stop_offset,
                                            current_stop_time_idx);

    if (CriteriaConfig::is_rescan_from_stop_needed(criteria_data,
                                                   trait_offset)) {
      // scan through the remaining trips scanning for a trip which gives
      //   a feasible arrival on this stop with this trait offset
      auto const [new_trip_id, new_trip_data] =
          get_next_feasible_trip<CriteriaConfig>(tt, prev_arrivals, r_id,
                                                 earliest_trip_id, trait_offset,
                                                 r_stop_offset);

      if (valid(new_trip_id)) {
        criteria_data = std::move(new_trip_data);
        earliest_trip_id = new_trip_id;
        current_stop_time_idx = route.index_to_stop_times_ +
                                (earliest_trip_id * route.stop_count_) +
                                r_stop_offset;
      } else {
        // no trip was found on this route which matches the trait_offset
        // while going from the known departure station we can also skip all
        // further stops until finding a new stop we can depart from
        CriteriaConfig::reset_traits_aggregate(criteria_data);
        earliest_trip_id = invalid<trip_count>;

        // it is still possible that this stop can serve as departure stop
        --r_stop_offset;
        continue;
      }
    }

    auto const& stop_time = tt.stop_times_[current_stop_time_idx];
    auto const stop_arr_idx =
        CriteriaConfig::get_arrival_idx(stop_id, trait_offset);

    // need the minimum due to footpaths updating arrivals
    // and not earliest arrivals
    //includes local and target pruning
    auto const ea_target_min = std::min(ea[stop_arr_idx], ea[target_arr_idx]);
    auto const min = std::min(current_round[stop_arr_idx], ea_target_min);

    if (stop_time.arrival_ < min &&
        CriteriaConfig::is_update_required(criteria_data, trait_offset)) {
      station_marks.mark(stop_id);
      current_round[stop_arr_idx] = stop_time.arrival_;
    }

    /*
     * The reason for the split in the update process for the current_round
     * and the earliest arrivals is that we might have some results in
     * current_round from former runs of the algorithm, but the earliest
     * arrivals start at invalid<time> every run.
     *
     * Therefore, we need to set the earliest arrival independently from
     * the results in current round.
     *
     * We cannot carry over the earliest arrivals from former runs, since
     * then we would skip on updates to the curren_round results.
     */

    if (stop_time.arrival_ < ea[stop_arr_idx] &&
        CriteriaConfig::is_update_required(criteria_data, trait_offset)) {
      ea[stop_arr_idx] = stop_time.arrival_;
    }

    // check if we could catch an earlier trip
    auto const previous_k_arrival = prev_arrivals[stop_arr_idx];
    if (valid(stop_time.departure_) &&
        previous_k_arrival <= stop_time.departure_) {
      earliest_trip_id =
          std::min(earliest_trip_id,
                   get_earliest_trip<CriteriaConfig>(
                       tt, route, prev_arrivals, r_stop_offset, trait_offset));

      CriteriaConfig::reset_traits_aggregate(criteria_data);
    }
  }
}

template <typename CriteriaConfig>
void update_route(raptor_timetable const& tt, route_id const r_id,
                  time const* const prev_arrivals, time* const current_round,
                  earliest_arrivals& ea, cpu_mark_store& station_marks,
                  stop_id const target_s_id) {
  auto const trait_size = CriteriaConfig::trait_size();
  for (uint32_t t_offset = 0; t_offset < trait_size; ++t_offset) {
    auto const target_arr_idx =
        CriteriaConfig::get_arrival_idx(target_s_id, t_offset);
    update_route_for_trait_offset<CriteriaConfig>(
        tt, r_id, prev_arrivals, current_round, ea, station_marks, t_offset,
        target_arr_idx);
  }
}

template <typename CriteriaConfig>
inline void update_footpaths(raptor_timetable const& tt, time* current_round,
                             earliest_arrivals& ea,
                             cpu_mark_store& station_marks) {

  // How far do we need to skip until the next stop is reached?
  auto const trait_size = CriteriaConfig::trait_size();

  for (stop_id stop_id = 0; stop_id < tt.stop_count(); ++stop_id) {

    auto index_into_transfers = tt.stops_[stop_id].index_to_transfers_;
    auto next_index_into_transfers = tt.stops_[stop_id + 1].index_to_transfers_;

    for (auto current_index = index_into_transfers;
         current_index < next_index_into_transfers; ++current_index) {

      auto const& footpath = tt.footpaths_[current_index];

      for (int s_trait_offset = 0; s_trait_offset < trait_size;
           ++s_trait_offset) {
        auto const from_arr_idx =
            CriteriaConfig::get_arrival_idx(stop_id, s_trait_offset);
        auto const to_arr_idx =
            CriteriaConfig::get_arrival_idx(footpath.to_, s_trait_offset);

        if (!valid(ea[from_arr_idx])) {
          continue;
        }

        // there is no triangle inequality in the footpath graph!
        // we cannot use the normal arrival values,
        // but need to use the earliest arrival values as read
        // and write to the normal arrivals,
        // otherwise it is possible that two footpaths
        // are chained together
        motis::time const new_arrival = ea[from_arr_idx] + footpath.duration_;

        motis::time to_arrival = current_round[to_arr_idx];
        motis::time to_earliest_arrival = ea[to_arr_idx];

        auto const min = std::min(to_arrival, to_earliest_arrival);
        if (new_arrival < min) {
          station_marks.mark(footpath.to_);
          current_round[to_arr_idx] = new_arrival;
        }
      }
    }
  }
}

template <typename CriteriaConfig>
void invoke_mc_cpu_raptor(const raptor_query& query, raptor_statistics&) {
  auto const& tt = query.tt_;
  auto& result = *query.result_;
  auto const target_s_id = query.target_;

#ifdef _DEBUG
  print_query(query);
  //    print_routes({9663}, tt);
  //     print_stations(raptor_sched);
  //     print_route_trip_debug_strings(raptor_sched);
#endif
  // print_route(15118, tt);
  //  print_routes(get_routes_containing(std::vector<int>{5210,15072}, tt),
  //  tt);

  earliest_arrivals ea(tt.stop_count() * CriteriaConfig::trait_size(),
                       invalid<motis::time>);

  earliest_arrivals current_round_arrivals(tt.stop_count() *
                                           CriteriaConfig::trait_size());

  cpu_mark_store station_marks(tt.stop_count());
  cpu_mark_store route_marks(tt.route_count());

  init_arrivals<CriteriaConfig>(result, query, station_marks);

  for (raptor_round round_k = 1; round_k < max_raptor_round; ++round_k) {
    bool any_marked = false;

    for (auto s_id = 0; s_id < tt.stop_count(); ++s_id) {
      if (!station_marks.marked(s_id)) {
        continue;
      }
      if (!any_marked) any_marked = true;
      auto const& stop = tt.stops_[s_id];
      for (auto sri = stop.index_to_stop_routes_;
           sri < stop.index_to_stop_routes_ + stop.route_count_; ++sri) {
        route_marks.mark(tt.stop_routes_[sri]);
      }
    }
    if (!any_marked) {
      break;
    }

    station_marks.reset();

    for (route_id r_id = 0; r_id < tt.route_count(); ++r_id) {
      if (!route_marks.marked(r_id)) {
        continue;
      }

      update_route<CriteriaConfig>(tt, r_id, result[round_k - 1],
                                   result[round_k], ea, station_marks,
                                   target_s_id);
    }

    // if(round_k == 2)
    //   print_results<CriteriaConfig>(result, tt, 3, 0);

    route_marks.reset();

    std::memcpy(current_round_arrivals.data(), result[round_k],
                current_round_arrivals.size() * sizeof(motis::time));

    update_footpaths<CriteriaConfig>(tt, result[round_k],
                                     current_round_arrivals, station_marks);
  }
}

RAPTOR_CRITERIA_CONFIGS_WO_DEFAULT(MAKE_MC_CPU_RAPTOR_TEMPLATE_INSTANCE, )

}  // namespace motis::raptor