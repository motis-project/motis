#include "motis/raptor/cpu/mc_cpu_raptor.h"

#include "motis/raptor/cpu/mark_store.h"
#include "motis/raptor/criteria/configs.h"
#include "motis/raptor/raptor_query.h"
#include "motis/raptor/raptor_result.h"
#include "motis/raptor/raptor_statistics.h"
#include "motis/raptor/raptor_timetable.h"

#include "motis/core/common/timing.h"

namespace motis::raptor {

template <typename CriteriaConfig>
inline void init_arrivals(raptor_result& result, raptor_query const& q,
                          raptor_timetable const& tt,
                          cpu_mark_store& station_marks) {

  auto const traits_size = CriteriaConfig::TRAITS_SIZE;
  auto const sweep_block_size = CriteriaConfig::SWEEP_BLOCK_SIZE;

  // init arrival times to the first index of every block along the
  // trait size

  auto propagate_across_traits = [&](time* const arrivals, stop_id stop_id,
                                     motis::time arrival_val) {
    auto const first_arr_idx = (stop_id * traits_size);
    auto const last_arr_idx = first_arr_idx + traits_size;

    for (arrival_id arr_idx = first_arr_idx; arr_idx < last_arr_idx;
         arr_idx += sweep_block_size) {
      auto const t_offset = arr_idx - first_arr_idx;
      time const arrival_time = arrival_val - CriteriaConfig::get_transfer_time(
                                                  tt, t_offset, stop_id);
      arrivals[arr_idx] = std::min(arrival_time, arrivals[arr_idx]);
      station_marks.mark(arr_idx);
    }
  };

  propagate_across_traits(result[0], q.source_, q.source_time_begin_);

  for (auto const& add_start : q.add_starts_) {
    time const add_start_time = q.source_time_begin_ + add_start.offset_;

    propagate_across_traits(result[0], add_start.s_id_, add_start_time);
  }
}

template <typename CriteriaConfig>
inline void update_route_for_trait_offset_forward_project(
    raptor_timetable const& tt, trait_id const trait_offset,
    route_id const r_id, time const* const previous_round,
    time* const current_round, earliest_arrivals& ea,
    cpu_mark_store& station_marks, stop_id const target_s_id,
    uint64_t const fp_offset, cpu_mark_store& fp_marks) {

  auto const& route = tt.routes_[r_id];
  CriteriaConfig aggregate{&route, trait_offset};

  auto active_stop_count = route.stop_count_;

  for (trip_count trip_id = 0; trip_id < route.trip_count_; ++trip_id) {
    aggregate.reset(trip_id, trait_offset);

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
        aggregate.update_from_stop(tt, r_stop_offset, current_sti);

        auto const write_off = aggregate.get_write_to_trait_id();
        if (valid(write_off)) {
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
            fp_marks.unmark(fp_offset + write_arr);
          }

          if (current_stop_time.arrival_ < ea[write_arr]) {
            ea[write_arr] = current_stop_time.arrival_;
          }

          if (aggregate.is_satisfied(trait_offset)) {
            // either we wrote a time or already know a better time
            ++consecutive_writes;
          } else {
            // we didn't satisfy; therefor at least this stop can still improve
            //  which is after stops which potentially received a value
            consecutive_writes = 0;
          }
        } else {
          departure_offset = invalid<stop_offset>;
          consecutive_writes = 0;
        }
      }

      // can station serve as departure station?
      if (valid(previous_round[arrival_idx]) &&
          valid(current_stop_time.departure_) &&
          aggregate.check_and_set_departure_stop(tt, r_stop_offset, stop_id,
                                                 previous_round[arrival_idx],
                                                 current_stop_time.departure_)
      ) {
        departure_offset = r_stop_offset;
        consecutive_writes = 0;
        continue;
      }
    }

    active_stop_count -= consecutive_writes;
    if (active_stop_count <= 1) break;
  }
}

template <typename CriteriaConfig>
inline void perform_arrival_sweeping(stop_id const stop_count,
                                     time* const current_round,
                                     cpu_mark_store& station_marks) {
  for (stop_id s_id = 0; s_id < stop_count; ++s_id) {
    CriteriaConfig::perform_stop_arrival_sweeping_cpu(s_id, current_round, station_marks);
  }
}

template <typename CriteriaConfig>
inline void update_footpaths(raptor_timetable const& tt, time* current_round,
                             earliest_arrivals const& current_round_arr_const,
                             earliest_arrivals& ea,
                             cpu_mark_store& station_marks,
                             stop_id const target_s_id,
                             uint64_t const fp_offset,
                             cpu_mark_store& fp_arrivals) {

  // How far do we need to skip until the next stop is reached?
  auto const trait_size = CriteriaConfig::TRAITS_SIZE;
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

        if (!valid(current_round_arr_const[from_arr_idx])
            || fp_arrivals.marked(fp_offset + from_arr_idx)) {
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

        // local pruning
        auto min = std::min(to_arrival, to_ea);
        // target pruning
        min = std::min(min, ea[target_arr_idx + s_trait_offset]);
        if (new_arrival < min) {
          station_marks.mark(to_arr_idx);
          current_round[to_arr_idx] = new_arrival;
          ea[to_arr_idx] = new_arrival;
          fp_arrivals.mark(fp_offset + to_arr_idx);
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

  auto const trait_size = CriteriaConfig::TRAITS_SIZE;
  earliest_arrivals ea(tt.stop_count() * trait_size, invalid<motis::time>);

  earliest_arrivals current_round_arrivals(tt.stop_count() * trait_size);

  cpu_mark_store station_marks(tt.stop_count() * trait_size);
  cpu_mark_store route_marks(tt.route_count() * trait_size);

  init_arrivals<CriteriaConfig>(result, query, tt, station_marks);

  for (raptor_round round_k = 1; round_k < max_raptor_round; ++round_k) {
    auto const round_offset = (round_k - 1) * tt.stop_count() * trait_size;
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

        update_route_for_trait_offset_forward_project<CriteriaConfig>(
            tt, t_offset, r_id, result[round_k - 1], result[round_k], ea,
            station_marks, target_s_id, round_offset, *query.fp_times_);
      }
    }
    auto const route_time = MOTIS_GET_TIMING_US(route_update);
    stats.cpu_time_routes_ += route_time;

    route_marks.reset();

    MOTIS_START_TIMING(prune_arrivals);
    perform_arrival_sweeping<CriteriaConfig>(tt.stop_count(), result[round_k],
                                             station_marks);
    auto const prune_time = MOTIS_GET_TIMING_US(prune_arrivals);
    stats.cpu_time_clear_arrivals_ += prune_time;

    MOTIS_START_TIMING(footpath_update);
    std::memcpy(current_round_arrivals.data(), result[round_k],
                current_round_arrivals.size() * sizeof(motis::time));

    update_footpaths<CriteriaConfig>(tt, result[round_k],
                                     current_round_arrivals, ea, station_marks,
                                     target_s_id, round_offset, *query.fp_times_);
    auto const fp_time = MOTIS_GET_TIMING_US(footpath_update);
    stats.cpu_time_footpath_ += fp_time;
    stats.number_of_rounds_ = round_k;
  }
}

RAPTOR_CRITERIA_CONFIGS_WO_DEFAULT(MAKE_MC_CPU_RAPTOR_TEMPLATE_INSTANCE, )

}  // namespace motis::raptor