#pragma once

#include <fstream>
#include <iostream>

#include "fmt/core.h"
#include "fmt/ostream.h"

#include "utl/repeat_n.h"

#include "motis/path/prepare/resolve/processing_plan.h"
#include "motis/path/prepare/schedule/station_sequences.h"
#include "motis/path/prepare/seq/seq_graph.h"

namespace motis::path {

inline void print_seq_graph(std::vector<part_task> const& part_tasks,
                            seq_graph const& g, station_seq const& seq) {
  static std::mutex m;
  std::lock_guard l{m};

  auto nodes = utl::repeat_n(std::vector<seq_node*>{}, g.seq_size_);
  for (auto& node : g.nodes_) {
    nodes[node->station_idx_].push_back(node.get());
  }

  std::ofstream out{"seq_graph.txt", std::ofstream::app};

  out << "\n---\n\n";
  for (auto i = 0UL; i < nodes.size(); ++i) {
    out << "station: " << i << " / " << seq.station_ids_.at(i) << " / "
        << seq.station_names_.at(i) << "\n";
    for (auto const& node : nodes[i]) {
      out << "  " << node->idx_ << " @ " << node->ref_.strategy_id_ << " -> ";
      for (auto const& edge : node->edges_) {
        auto strategy_id =
            edge.part_task_idx_ == kInvalidPartTask
                ? 0
                : part_tasks[edge.part_task_idx_].key_.strategy_->strategy_id_;

        fmt::print(out, "{}:{}|{}:{:.1e} ", edge.to_->ref_.strategy_id_,
                   edge.to_->idx_, strategy_id, edge.weight());
      }
      out << "\n";
    }
  }
}

}  // namespace motis::path
