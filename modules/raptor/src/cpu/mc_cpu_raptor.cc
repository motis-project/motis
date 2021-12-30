#include "motis/raptor/cpu/mc_cpu_raptor.h"

#include "motis/raptor/cpu/mark_store.h"
#include "motis/raptor/criteria/configs.h"
#include "motis/raptor/print_raptor.h"
#include "motis/raptor/raptor_query.h"
#include "motis/raptor/raptor_result.h"
#include "motis/raptor/raptor_statistics.h"
#include "motis/raptor/raptor_timetable.h"

#include "motis/core/common/timing.h"

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
                          cpu_mark_store& station_marks, bool const propagate) {

  auto const traits_size = CriteriaConfig::trait_size();
  auto propagate_across_traits = [traits_size, &station_marks](
                                     time* const arrivals, stop_id stop_id,
                                     motis::time arrival_val) {
    auto const last_arr_idx = (stop_id * traits_size) + traits_size;
    for (arrival_id arr_idx = (stop_id * traits_size); arr_idx < last_arr_idx;
         ++arr_idx) {
      arrivals[arr_idx] = std::min(arrival_val, arrivals[arr_idx]);
      station_marks.mark(arr_idx);
    }
  };

  result[0][traits_size * q.source_] = q.source_time_begin_;
  if (propagate)
    propagate_across_traits(result[0], q.source_, q.source_time_begin_);
  station_marks.mark(q.source_ * traits_size);

  for (auto const& add_start : q.add_starts_) {
    motis::time const add_start_time = q.source_time_begin_ + add_start.offset_;
    result[0][traits_size * add_start.s_id_] =
        std::min(result[0][traits_size * add_start.s_id_], add_start_time);
    station_marks.mark(add_start.s_id_ * traits_size);
    if (propagate) {
      propagate_across_traits(result[0], add_start.s_id_, add_start_time);
    }
  }
}

template <typename CriteriaConfig>
inline std::tuple<trip_id, typename CriteriaConfig::CriteriaData>
get_next_feasible_trip(raptor_timetable const& tt,
                       time const* const prev_arrivals, route_id r_id,
                       trip_id earliest_trip_id, trait_id trait_offset,
                       stop_offset arr_offset) {

  auto const& route = tt.routes_[r_id];
  stop_offset new_dep_offset;

  typename CriteriaConfig::CriteriaData trip_data{};

  for (trip_id trip_id = earliest_trip_id + 1; trip_id < route.trip_count_;
       ++trip_id) {
    CriteriaConfig::reset_traits_aggregate(trip_data, r_id, trip_id,
                                           trait_offset);

    // aggregate the trait data for the current departure station
    new_dep_offset = arr_offset - 1;
    stop_times_index const arr_sti =
        route.index_to_stop_times_ + (trip_id * route.stop_count_) + arr_offset;
    CriteriaConfig::update_traits_aggregate(trip_data, tt, prev_arrivals,
                                            arr_offset, arr_sti);

    do {
      auto const dep_sti = route.index_to_stop_times_ +
                           (trip_id * route.stop_count_) + new_dep_offset;
      auto const dep_s_id =
          tt.route_stops_[route.index_to_route_stops_ + new_dep_offset];
      auto const dep_arr_idx =
          CriteriaConfig::get_arrival_idx(dep_s_id, trait_offset);
      auto dep_stop_times = tt.stop_times_[dep_sti];

      if (CriteriaConfig::get_write_to_trait_id(trip_data) == trait_offset &&
          valid(dep_stop_times.departure_) &&
          prev_arrivals[dep_arr_idx] <= dep_stop_times.departure_) {
        return std::make_tuple(trip_id, std::move(trip_data));
      }

      CriteriaConfig::update_traits_aggregate(trip_data, tt, prev_arrivals,
                                              new_dep_offset, dep_sti);
      --new_dep_offset;

    } while (new_dep_offset != 0 &&
             CriteriaConfig::get_write_to_trait_id(trip_data) == trait_offset);
  }

  return std::make_tuple(invalid<raptor::trip_id>, std::move(trip_data));
}

template <typename CriteriaConfig>
void update_route_for_trait_offset(
    raptor_timetable const& tt, route_id const r_id,
    time const* const prev_arrivals, time* const current_round,
    earliest_arrivals& ea, cpu_mark_store& station_marks, trip_id trait_offset,
    arrival_id target_arr_idx) {
  auto const& route = tt.routes_[r_id];

  typename CriteriaConfig::CriteriaData criteria_data{};

  trip_count earliest_trip_id = invalid<trip_count>;
  for (stop_id r_stop_offset = 0; r_stop_offset < route.stop_count_;
       ++r_stop_offset) {

    if (!valid(earliest_trip_id)) {
      earliest_trip_id = get_earliest_trip<CriteriaConfig>(
          tt, route, prev_arrivals, r_stop_offset, trait_offset);

      CriteriaConfig::reset_traits_aggregate(criteria_data, r_id,
                                             earliest_trip_id, trait_offset);
      continue;
    }

    auto const stop_id =
        tt.route_stops_[route.index_to_route_stops_ + r_stop_offset];
    auto current_stop_time_idx = route.index_to_stop_times_ +
                                 (earliest_trip_id * route.stop_count_) +
                                 r_stop_offset;

    CriteriaConfig::update_traits_aggregate(
        criteria_data, tt, prev_arrivals, r_stop_offset, current_stop_time_idx);

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
        earliest_trip_id = invalid<trip_count>;
        CriteriaConfig::reset_traits_aggregate(criteria_data, r_id,
                                               earliest_trip_id, trait_offset);

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
    // includes local and target pruning
    auto const ea_target_min = std::min(ea[stop_arr_idx], ea[target_arr_idx]);
    auto const min = std::min(current_round[stop_arr_idx], ea_target_min);

    if (stop_time.arrival_ < min &&
        CriteriaConfig::get_write_to_trait_id(criteria_data) == trait_offset) {
      station_marks.mark(stop_arr_idx);
      current_round[stop_arr_idx] = stop_time.arrival_;
    }

    /*
     * The reason for the split in the update process for the current_round
     * and the earliest arrivals is that we might have some results in
     * current_round from former runs of the algorithm, but the earliest
     * arrivals start at invalid<time> every run.
     *
     * Therefore, we need to set the earliest arrival independently of
     * the results in current round.
     *
     * We cannot carry over the earliest arrivals from former runs, since
     * then we would skip on updates to the curren_round results.
     */

    if (stop_time.arrival_ < ea[stop_arr_idx] &&
        CriteriaConfig::get_write_to_trait_id(criteria_data) == trait_offset) {
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

      CriteriaConfig::reset_traits_aggregate(criteria_data, r_id,
                                             earliest_trip_id, trait_offset);
    }
  }
}

template <typename CriteriaConfig>
inline void update_route_for_trait_offset_forward_project(
    raptor_timetable const& tt, trait_id const trait_offset,
    route_id const r_id, time const* const previous_round,
    time* const current_round, earliest_arrivals& ea,
    cpu_mark_store& station_marks, stop_id const target_s_id) {

  auto const& route = tt.routes_[r_id];

  typename CriteriaConfig::CriteriaData aggregate{};

  auto active_stop_count = route.stop_count_;

  for (trip_count trip_id = 0; trip_id < route.trip_count_; ++trip_id) {
    CriteriaConfig::reset_traits_aggregate(aggregate, r_id, trip_id,
                                           trait_offset);

    auto const trip_first_sti =
        route.index_to_stop_times_ + (trip_id * route.stop_count_);

    stop_id departure_offset = invalid<stop_id>;
    auto consecutive_writes = 0;

    for (stop_id r_stop_offset = 0; r_stop_offset < active_stop_count;
         ++r_stop_offset) {

      stop_id const stop_id =
          tt.route_stops_[route.index_to_route_stops_ + r_stop_offset];

      auto const current_sti = trip_first_sti + r_stop_offset;

      auto const current_stop_time = tt.stop_times_[current_sti];
      auto const arrival_idx =
          CriteriaConfig::get_arrival_idx(stop_id, trait_offset);

      // it's important to first check if a better arrival time can be archived
      //  before checking if the station can serve as departure station
      //  otherwise potentially improved arrival times are not written
      if (valid(departure_offset)) {
        CriteriaConfig::update_traits_aggregate(aggregate, tt, previous_round,
                                                r_stop_offset, current_sti);

        auto const write_off = CriteriaConfig::get_write_to_trait_id(aggregate);
        auto const write_arr =
            CriteriaConfig::get_arrival_idx(stop_id, write_off);
        auto const earl_arr = ea[write_arr];
        auto const target_arr =
            CriteriaConfig::get_arrival_idx(target_s_id, write_off);
        auto const earl_tar = ea[target_arr];

        auto min_ea = std::min(earl_arr, earl_tar);
        min_ea = std::min(min_ea, current_round[write_arr]);

        if (valid(current_stop_time.arrival_) &&
            current_stop_time.arrival_ < min_ea) {
          current_round[write_arr] = current_stop_time.arrival_;
          station_marks.mark(write_arr);
        }

        if (CriteriaConfig::is_trait_satisfied(aggregate, trait_offset)) {
          // either we wrote a time or already know a better time
          ++consecutive_writes;
        } else {
          // we didn't satisfy; therefor at least this stop can still improve
          //  which is after stops which potentially received a value
          consecutive_writes = 0;
        }

        if (current_stop_time.arrival_ < ea[write_arr]) {
          ea[write_arr] = current_stop_time.arrival_;
        }
      }

      // can station serve as departure station?
      if (valid(previous_round[arrival_idx]) &&
          valid(current_stop_time.departure_) &&
          previous_round[arrival_idx] <= current_stop_time.departure_) {
        departure_offset = r_stop_offset;
        CriteriaConfig::reset_traits_aggregate(aggregate, r_id, trip_id,
                                               trait_offset);
        consecutive_writes = 0;
        continue;
      }
    }

    active_stop_count -= consecutive_writes;
    if (active_stop_count <= 1) break;
  }
}

template <typename CriteriaConfig>
inline void clear_dominated_arrivals(stop_id const stop_count,
                                     time* const current_round,
                                     earliest_arrivals& ea,
                                     cpu_mark_store& station_marks) {
  auto const trait_size = CriteriaConfig::trait_size();
  for (stop_id s_id = 0; s_id < stop_count; ++s_id) {
    time min_arrival_at_stop = current_round[trait_size * s_id];
    for (trait_id t_offset = 1; t_offset < trait_size; ++t_offset) {
      // if the value is larger or equal than the minimum we can prune it
      //   because it is dominated by the minimum on the earliest trait offset
      if (min_arrival_at_stop <= current_round[trait_size * s_id + t_offset]) {
        current_round[trait_size * s_id + t_offset] = invalid<time>;
        station_marks.unmark(trait_size * s_id + t_offset);
        if(min_arrival_at_stop < ea[trait_size * s_id + t_offset])
          ea[trait_size * s_id + t_offset] = min_arrival_at_stop;
      } else {
        // a higher t_offset has a better value; remember the larger value
        //  to again check higher t_offsets against it
        min_arrival_at_stop = current_round[trait_size * s_id + t_offset];
      }
    }
  }
}

template <typename CriteriaConfig>
inline void update_footpaths(raptor_timetable const& tt, time* current_round,
                             earliest_arrivals const& current_round_arr_const,
                             earliest_arrivals& ea,
                             cpu_mark_store& station_marks,
                             stop_id const target_s_id) {

  // How far do we need to skip until the next stop is reached?
  auto const trait_size = CriteriaConfig::trait_size();
  auto const target_arr_idx = CriteriaConfig::get_arrival_idx(target_s_id);

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

        if (!valid(current_round_arr_const[from_arr_idx])) {
          continue;
        }

        // there is no triangle inequality in the footpath graph!
        // we cannot use the normal arrival values,
        // but need to use the earliest arrival values as read
        // and write to the normal arrivals,
        // otherwise it is possible that two footpaths
        // are chained together
        motis::time const new_arrival =
            current_round_arr_const[from_arr_idx] + footpath.duration_;

        motis::time to_arrival = current_round[to_arr_idx];
        motis::time to_ea = ea[to_arr_idx];

        auto min = std::min(to_arrival, to_ea);  // local pruning
        min = std::min(min,
                       ea[target_arr_idx + s_trait_offset]);  // target pruning
        if (new_arrival < min) {
          station_marks.mark(to_arr_idx);
          current_round[to_arr_idx] = new_arrival;
          ea[to_arr_idx] = new_arrival;
        }
      }
    }
  }
}

template <typename CriteriaConfig>
void invoke_mc_cpu_raptor(const raptor_query& query, raptor_statistics& stats) {
  auto const& tt = query.tt_;
  auto& result = *query.result_;
  auto const target_s_id = query.target_;

  auto const use_fwd_prop = CriteriaConfig::is_forward_propagation_required();

  auto const trait_size = CriteriaConfig::trait_size();
  earliest_arrivals ea(tt.stop_count() * trait_size, invalid<motis::time>);

  earliest_arrivals current_round_arrivals(tt.stop_count() * trait_size);

  cpu_mark_store station_marks(tt.stop_count() * trait_size);
  cpu_mark_store route_marks(tt.route_count() * trait_size);

  init_arrivals<CriteriaConfig>(result, query, station_marks, !use_fwd_prop);

  for (raptor_round round_k = 1; round_k < max_raptor_round; ++round_k) {
    bool any_marked = false;

    for (auto s_id = 0; s_id < tt.stop_count(); ++s_id) {
      for (auto t_offset = 0; t_offset < trait_size; ++t_offset) {
        if (!station_marks.marked(s_id * trait_size + t_offset)) {
          continue;
        }
        if (!any_marked) any_marked = true;
        auto const& stop = tt.stops_[s_id];
        for (auto sri = stop.index_to_stop_routes_;
             sri < stop.index_to_stop_routes_ + stop.route_count_; ++sri) {
          route_marks.mark(tt.stop_routes_[sri] * trait_size + t_offset);
        }
      }
    }
    if (!any_marked) {
      break;
    }

    station_marks.reset();

    MOTIS_START_TIMING(route_update);
    for (uint32_t t_offset = 0; t_offset < trait_size; ++t_offset) {
      for (route_id r_id = 0; r_id < tt.route_count(); ++r_id) {
        if (!route_marks.marked(r_id * trait_size + t_offset)) {
          continue;
        }

        if (use_fwd_prop) {
          update_route_for_trait_offset_forward_project<CriteriaConfig>(
              tt, t_offset, r_id, result[round_k - 1], result[round_k], ea,
              station_marks, target_s_id);
        } else {
          auto const target_arr_idx =
              CriteriaConfig::get_arrival_idx(target_s_id, t_offset);
          update_route_for_trait_offset<CriteriaConfig>(
              tt, r_id, result[round_k - 1], result[round_k], ea, station_marks,
              t_offset, target_arr_idx);
        }
      }
    }
    auto const route_time = MOTIS_GET_TIMING_US(route_update);
    stats.cpu_time_routes_ += route_time;

    route_marks.reset();

    MOTIS_START_TIMING(prune_arrivals);
    clear_dominated_arrivals<CriteriaConfig>(tt.stop_count(), result[round_k],
                                             ea, station_marks);
    auto const prune_time = MOTIS_GET_TIMING_US(prune_arrivals);
    stats.cpu_time_clear_arrivals_ += prune_time;

    MOTIS_START_TIMING(footpath_update);
    std::memcpy(current_round_arrivals.data(), result[round_k],
                current_round_arrivals.size() * sizeof(motis::time));

    update_footpaths<CriteriaConfig>(tt, result[round_k],
                                     current_round_arrivals, ea, station_marks,
                                     target_s_id);
    auto const fp_time = MOTIS_GET_TIMING_US(footpath_update);
    stats.cpu_time_footpath_ += fp_time;
  }
}

RAPTOR_CRITERIA_CONFIGS_WO_DEFAULT(MAKE_MC_CPU_RAPTOR_TEMPLATE_INSTANCE, )

}  // namespace motis::raptor