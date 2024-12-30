#pragma once

#include <vector>

#include "nigiri/routing/query.h"

namespace motis::odm {

struct query_factory {
  std::vector<nigiri::routing::query> get_meta_routing_queries() const;

  // invariants
  nigiri::routing::start_time_t start_time_;
  nigiri::routing::location_match_mode start_match_mode_;
  nigiri::routing::location_match_mode dest_match_mode_;
  bool use_start_footpaths_;
  std::uint8_t max_transfers_;
  nigiri::duration_t max_travel_time_;
  unsigned min_connection_count_;
  bool extend_interval_earlier_;
  bool extend_interval_later_;
  nigiri::profile_idx_t prf_idx_;
  nigiri::routing::clasz_mask_t allowed_claszes_;
  bool require_bike_transport_;
  nigiri::routing::transfer_time_settings transfer_time_settings_;
  std::vector<nigiri::routing::via_stop> via_stops_;
  std::optional<nigiri::duration_t> fastest_direct_;

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
      odm_from_short_;
  nigiri::hash_map<nigiri::location_idx_t,
                   std::vector<nigiri::routing::td_offset>>
      odm_from_long_;
  nigiri::hash_map<nigiri::location_idx_t,
                   std::vector<nigiri::routing::td_offset>>
      odm_to_short_;
  nigiri::hash_map<nigiri::location_idx_t,
                   std::vector<nigiri::routing::td_offset>>
      odm_to_long_;

private:
  nigiri::routing::query walk_walk() const;
  nigiri::routing::query walk_short() const;
  nigiri::routing::query walk_long() const;
  nigiri::routing::query short_walk() const;
  nigiri::routing::query long_walk() const;
  nigiri::routing::query short_short() const;
  nigiri::routing::query short_long() const;
  nigiri::routing::query long_short() const;
  nigiri::routing::query long_long() const;
};

}  // namespace motis::odm