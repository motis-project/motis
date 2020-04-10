#pragma once

#include <limits>
#include <memory>
#include <vector>

#include "geo/latlng.h"
#include "geo/polyline.h"

#include "motis/path/prepare/source_spec.h"
#include "motis/path/prepare/strategy/routing_strategy.h"

namespace motis::path {

struct seq_edge;

struct seq_node {
  seq_node(size_t const station_idx, node_ref const& ref)
      : idx_(0),  // only assigned in finalize
        incomming_edges_count_(0),
        station_idx_(station_idx),
        ref_(ref) {}

  size_t strategy_id() const { return ref_.strategy_id_; }

  size_t idx_;

  std::vector<seq_edge> edges_;
  size_t incomming_edges_count_;

  size_t station_idx_;
  node_ref ref_;
};

struct seq_edge {
  seq_edge(seq_node* from, seq_node* to, size_t part_task_idx, double weight)
      : from_(from), to_(to), part_task_idx_{part_task_idx}, weight_(weight) {}

  double weight() const { return weight_; }

  seq_node* from_;
  seq_node* to_;

  size_t part_task_idx_;

  double weight_;
};

struct seq_graph {
  explicit seq_graph(size_t const seq_size) : seq_size_(seq_size) {}

  size_t seq_size_;
  std::vector<std::unique_ptr<seq_node>> nodes_;
  std::vector<std::pair<seq_node*, seq_node*>> node_pairs_;

  std::vector<std::size_t> initials_;
  std::vector<std::size_t> goals_;
};

}  // namespace motis::path
