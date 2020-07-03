#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "motis/core/common/hash_helper.h"

#include "motis/core/schedule/connection.h"

namespace motis::path {

struct path_routing;
struct routing_strategy;
struct station_seq;

using seq_task_idx_t = uint32_t;
using part_task_idx_t = uint32_t;

constexpr auto const kInvalidPartTask =
    std::numeric_limits<part_task_idx_t>::max();

struct seq_task {
  seq_task(station_seq const* seq, mcd::vector<service_class> classes)
      : seq_{seq}, classes_{std::move(classes)} {}

  station_seq const* seq_;
  mcd::vector<service_class> classes_;

  std::vector<part_task_idx_t> part_dependencies_;
};

struct part_task_key {
  routing_strategy* strategy_ = nullptr;
  std::string station_id_from_, station_id_to_;
};

struct part_task {
  part_task(uint32_t location_hash, part_task_key key)
      : location_hash_{location_hash}, key_{std::move(key)} {};

  uint32_t location_hash_;

  part_task_key key_;
  std::vector<seq_task_idx_t> seq_dependencies_;
};

struct processing_plan {
  std::vector<seq_task> seq_tasks_;
  std::vector<part_task> part_tasks_;

  std::vector<part_task_idx_t> part_task_queue_;
};

processing_plan make_processing_plan(path_routing&,
                                     mcd::vector<station_seq> const&);

}  // namespace motis::path