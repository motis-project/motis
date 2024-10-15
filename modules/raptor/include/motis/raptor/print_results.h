#pragma once

#include <iostream>

#include "motis/raptor/raptor_query.h"
#include "motis/raptor/raptor_result.h"
#include "motis/raptor/raptor_timetable.h"

namespace motis::raptor {

void print_results(raptor_result_base const& result,
                   raptor_meta_info const& meta,
                   raptor_round up_to_round = max_raptor_round,
                   uint32_t only_for_t_offset = invalid<uint32_t>) {
  auto const trait_size =
      get_trait_size_for_criteria_config(result.criteria_config_);
  auto trait_loop_start = 0;
  auto trait_loo_end = trait_size;

  if (valid(only_for_t_offset)) {
    trait_loo_end = only_for_t_offset + 1;
    trait_loop_start = only_for_t_offset;
  }

  try {

    for (raptor_round round_k = 0;
         round_k < max_raptor_round && round_k < up_to_round; ++round_k) {
      std::cerr << "==========================================================\n"
          << "Results Round " << +round_k << "\n"
          << "=========================================================="
          << std::endl;
      for (int i = 0; i < (result.arrival_times_count_ / trait_size) - 1; ++i) {
        auto const eva = meta.raptor_id_to_eva_.at(i);
        auto had_valid_time = false;
        for (int j = trait_loop_start; j < trait_loo_end; ++j) {
          if (valid(result[round_k][(i * trait_size) + j])) {

            if (!had_valid_time) {
              std::cerr << "Stop Id: " << std::setw(7) << +i << "(" << eva
                        << ") -> ";
              had_valid_time = true;
            }

            std::cerr << std::setw(6) << +result[round_k][(i * trait_size) + j]
                      << "; Arrival Idx: " << std::setw(6)
                      << +((i * trait_size) + j)
                      << "; Trait Offset: " << std::setw(4) << +j << ";\t\t";
          }
        }

        if (had_valid_time) std::cerr << std::endl;
      }
    }
  } catch (std::exception const& e) {
    std::cerr << "Catched" << std::endl;
  }

  std::cerr << std::endl;
}

template <typename Query>
void print_results_of_query(Query const& q, raptor_meta_info const& meta) {
  std::cout << "Called Default Impl!\n";
}

template <>
void print_results_of_query<raptor_query>(raptor_query const& q,
                                          raptor_meta_info const& meta) {
  raptor_result_base const& result = *q.result_;
  print_results(result, meta, 3);
}

template <>
void print_results_of_query<d_query>(d_query const& q,
                                     raptor_meta_info const& meta) {
  auto const& result = q.result();
  print_results(result, meta, 3);
}

}