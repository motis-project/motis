#pragma once

#include <vector>

#include "nigiri/routing/query.h"

namespace motis::odm {

struct query_factory {
  static constexpr auto const kMaxSubQueries = 9U;

  std::vector<nigiri::routing::query> make_queries(
      bool with_taxi, bool with_ride_sharing) const;

private:
  nigiri::routing::query make(
      std::vector<nigiri::routing::offset> const& start,
      nigiri::hash_map<nigiri::location_idx_t,
                       std::vector<nigiri::routing::td_offset>> const& td_start,
      std::vector<nigiri::routing::offset> const& dest,
      nigiri::hash_map<nigiri::location_idx_t,
                       std::vector<nigiri::routing::td_offset>> const& td_dest)
      const;

public:
  // invariants
  nigiri::routing::query base_query_;

  // offsets
  std::vector<nigiri::routing::offset> start_walk_;
  std::vector<nigiri::routing::offset> dest_walk_;
  nigiri::hash_map<nigiri::location_idx_t,
                   std::vector<nigiri::routing::td_offset>>
      td_start_walk_;
  nigiri::hash_map<nigiri::location_idx_t,
                   std::vector<nigiri::routing::td_offset>>
      td_dest_walk_;
  nigiri::hash_map<nigiri::location_idx_t,
                   std::vector<nigiri::routing::td_offset>>
      start_taxi_short_;
  nigiri::hash_map<nigiri::location_idx_t,
                   std::vector<nigiri::routing::td_offset>>
      start_taxi_long_;
  nigiri::hash_map<nigiri::location_idx_t,
                   std::vector<nigiri::routing::td_offset>>
      dest_taxi_short_;
  nigiri::hash_map<nigiri::location_idx_t,
                   std::vector<nigiri::routing::td_offset>>
      dest_taxi_long_;
  nigiri::hash_map<nigiri::location_idx_t,
                   std::vector<nigiri::routing::td_offset>>
      start_ride_sharing_;
  nigiri::hash_map<nigiri::location_idx_t,
                   std::vector<nigiri::routing::td_offset>>
      dest_ride_sharing_;
};

}  // namespace motis::odm