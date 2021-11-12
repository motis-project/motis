#include "motis/raptor/cpu/mc_cpu_raptor.h"

#include "motis/raptor/cpu/mark_store.h"
#include "motis/raptor/print_raptor.h"
#include "motis/raptor/raptor_query.h"
#include "motis/raptor/raptor_result.h"
#include "motis/raptor/raptor_statistics.h"
#include "motis/raptor/raptor_timetable.h"

namespace motis::raptor {

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
inline bool update_stop_if_required(
    typename CriteriaConfig::CriteriaData const& trait_data,
    uint32_t trait_offset, time* const current_round,
    cpu_mark_store& station_marks, earliest_arrivals& ea,
    stop_times_index const arrival_idx, stop_id stop_id,
    stop_time const& current_stop_time) {

  auto satisfied = false;
  if (CriteriaConfig::is_update_required(trait_data, trait_offset)) {

    auto const min = std::min(current_round[arrival_idx], ea[arrival_idx]);

    if (current_stop_time.arrival_ < min) {
      current_round[arrival_idx] = current_stop_time.arrival_;
      station_marks.mark(stop_id);
      satisfied = CriteriaConfig::is_trait_satisfied(trait_data, trait_offset);
    }

    if (current_stop_time.arrival_ < ea[arrival_idx]) {
      ea[arrival_idx] = current_stop_time.arrival_;
      // write the earliest arrival time for this stop after this round
      //  as this is a lower bound for the trip search
    }
  }

  return satisfied;
}

template <typename CriteriaConfig>
inline void update_route(raptor_timetable const& tt, route_id const r_id,
                         time const* const previous_round,
                         time* const current_round, earliest_arrivals& ea,
                         cpu_mark_store& station_marks) {

  auto const& route = tt.routes_[r_id];

  auto const trait_size = CriteriaConfig::trait_size();
  uint32_t satisfied_stop_cnt = 0;
  typename CriteriaConfig::CriteriaData trait_data{};

  for (trip_count trip_id = 0; trip_id < route.trip_count_; ++trip_id) {

    auto const trip_first_stop_sti =
        route.index_to_stop_times_ + (trip_id * route.stop_count_);

    for (uint32_t trait_offset = 0; trait_offset < trait_size; ++trait_offset) {
      stop_id departure_station = invalid<stop_id>;

      for (stop_id r_stop_offset = 0; r_stop_offset < route.stop_count_;
           ++r_stop_offset) {

        stop_id const stop_id =
            tt.route_stops_[route.index_to_route_stops_ + r_stop_offset];

        auto const current_sti = trip_first_stop_sti + r_stop_offset;

        auto const current_stop_time = tt.stop_times_[current_sti];
        auto const arrival_idx =
            CriteriaConfig::get_arrival_idx(stop_id, trait_offset);

        // iff there is an invalid departure id
        //     => we can skip if there is no arrival known at this stop
        //        or if the entering at this stop is not allowed
        //        or if the trip can't be caught at this stop
        if (!valid(departure_station) &&
            (!valid(previous_round[arrival_idx]) ||
             !valid(current_stop_time.departure_) ||
             previous_round[arrival_idx] > current_stop_time.departure_)) {
          continue;
        }

        if (valid(departure_station)) {
          // TODO: recheck whether this is obsolete
          CriteriaConfig::update_traits_aggregate(trait_data, tt, r_id, trip_id,
                                                  r_stop_offset, current_sti);

          // even though the current station could soon serve as departure
          // station
          //  it may still be that the arrival time improves
          //  for connections with the same trait offset but more transfers
          //  therefore we also want to store an arrival time for this station
          //  before it becomes the new departure station
          update_stop_if_required<CriteriaConfig>(
              trait_data, trait_offset, current_round, station_marks, ea,
              arrival_idx, stop_id, current_stop_time);
        }

        // if we could reach this stop in the previous round
        //   and the stop arrival time is earlier than the trip departure time
        //   this stop can serve as new departure stop
        //   as a departure later in the route can't worsen but just improve
        //   the result at the following stations it is preferred to reset
        //   the departure stop in these cases
        if (valid(current_stop_time.departure_) &&
            previous_round[arrival_idx] <= current_stop_time.departure_) {

          departure_station = stop_id;
          CriteriaConfig::reset_traits_aggregate(trait_data);

          // we can't improve the arrival time on the station the trip was
          // boarded
          continue;
        }

        update_stop_if_required<CriteriaConfig>(
            trait_data, trait_offset, current_round, station_marks, ea,
            arrival_idx, stop_id, current_stop_time);
      }

      CriteriaConfig::reset_traits_aggregate(trait_data);
    }

    if (satisfied_stop_cnt == route.stop_count_ - 1) {
      // we can't reach satisfaction for the first stop
      break;
    }
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

#ifdef _DEBUG
  print_query(query);
  // print_routes(std::vector<route_id>{98}, tt);
  // print_routes({9663}, tt);
  //  print_stations(raptor_sched);
  //  print_route_trip_debug_strings(raptor_sched);
#endif

  earliest_arrivals ea(tt.stop_count() * CriteriaConfig::trait_size(),
                       invalid<motis::time>);

  std::vector<motis::time> current_round_arrivals(tt.stop_count() *
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
                                   result[round_k], ea, station_marks);
    }

    route_marks.reset();

    update_footpaths<CriteriaConfig>(tt, result[round_k], ea, station_marks);
  }

  // print_results<CriteriaConfig>(result, tt, 6, 1);
}

template void invoke_mc_cpu_raptor<MaxOccupancy>(const raptor_query& query,
                                                 raptor_statistics&);

}  // namespace motis::raptor