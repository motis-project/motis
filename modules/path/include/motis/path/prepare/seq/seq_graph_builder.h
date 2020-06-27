#pragma once

#include <vector>

#include "utl/to_vec.h"

#include "motis/path/prepare/schedule/station_sequences.h"
#include "motis/path/prepare/seq/seq_graph.h"
#include "motis/path/prepare/strategy/routing_strategy.h"

namespace motis::path {

struct seq_graph_builder {
  explicit seq_graph_builder(station_seq const& seq)
      : seq_{seq},
        graph_{seq.station_ids_.size()},
        station_arrivals_(seq.station_ids_.size()),
        station_departures_(seq.station_ids_.size()) {}

  void add_part(size_t part_task_idx, routing_strategy const* strategy,
                std::string const& station_id_from,
                std::string const& station_id_to,
                routing_result_matrix const& routing_result) {
    if (curr_strategy_ != strategy) {
      utl::verify(curr_station_idx_ + 2 == seq_.station_ids_.size() ||
                      curr_station_idx_ == std::numeric_limits<size_t>::max(),
                  "invalid index");
      utl::verify(curr_at_station_ == false, "sequence ended at station");

      curr_strategy_ = strategy;
      curr_station_idx_ = 0;
      curr_at_station_ = false;
    } else {
      curr_at_station_ = !curr_at_station_;
      // advance if now at station
      curr_station_idx_ += static_cast<size_t>(curr_at_station_);
    }

    if (curr_at_station_) {
      utl::verify(seq_.station_ids_.at(curr_station_idx_) == station_id_from,
                  "seq_graph_builder: unexpected from station (at_station)");
      utl::verify(seq_.station_ids_.at(curr_station_idx_) == station_id_to,
                  "seq_graph_builder: unexpected to station (at_station)");
    } else {
      utl::verify(
          seq_.station_ids_.at(curr_station_idx_) == station_id_from,
          "seq_graph_builder: unexpected from station (not at_station)");
      utl::verify(seq_.station_ids_.at(curr_station_idx_ + 1) == station_id_to,
                  "seq_graph_builder: unexpected to station (not at_station)");
    }

    utl::verify(strategy != nullptr, "invalid strategy");  // clang-tidy ...
    if (curr_station_idx_ == 0) {
      from_nodes_ = make_nodes(strategy, station_id_from, curr_station_idx_);
    }
    auto to_nodes =
        make_nodes(strategy, station_id_to,
                   curr_at_station_ ? curr_station_idx_  // dep_i
                                    : curr_station_idx_ + 1  // arr_i+1
        );

    make_edges(from_nodes_, to_nodes, part_task_idx, routing_result);

    if (!curr_at_station_) {
      utl::concat(station_departures_.at(curr_station_idx_), from_nodes_);
      utl::concat(station_arrivals_.at(curr_station_idx_ + 1), to_nodes);
    }

    from_nodes_ = std::move(to_nodes);
  }

  std::vector<seq_node*> make_nodes(routing_strategy const* s,
                                    std::string const& station_id,
                                    size_t const station_idx) {
    return utl::to_vec(s->close_nodes(station_id), [&, this](auto const& ref) {
      return graph_.nodes_
          .emplace_back(std::make_unique<seq_node>(station_idx, ref))
          .get();
    });
  }

  static void make_edges(std::vector<seq_node*> const& from_nodes,
                         std::vector<seq_node*> const& to_nodes,  //
                         size_t part_task_idx,
                         routing_result_matrix const& routing_result) {
    routing_result.verify_dimensions(from_nodes.size(), to_nodes.size());
    routing_result.foreach (
        [&](auto const from_idx, auto const to_idx, auto const weight) {
          if (weight == std::numeric_limits<double>::infinity()) {
            return;
          }

          auto* from = from_nodes[from_idx];
          auto* to = to_nodes[to_idx];

          if (from->station_idx_ == to->station_idx_) {
            auto const dist = std::max(from->ref_.dist_, to->ref_.dist_);
            add_edge(from, to, part_task_idx, (weight + dist) * 100);
          } else {
            add_edge(from, to, part_task_idx, weight);
          }
        });
  }

  static void add_edge(seq_node* from, seq_node* to, size_t part_task_idx,
                       double const weight) {
    from->edges_.emplace_back(from, to, part_task_idx, weight);
    ++to->incomming_edges_count_;
  };

  void add_stub_edges(routing_strategy const* stub_strategy) {
    curr_strategy_ = stub_strategy;

    for (auto i = 0UL; i < seq_.station_ids_.size() - 1; ++i) {  // departures
      auto ref =
          stub_strategy->close_nodes(seq_.station_ids_.at(i).str()).front();
      station_departures_.at(i).push_back(
          graph_.nodes_.emplace_back(std::make_unique<seq_node>(i, ref)).get());
    }
    for (auto i = 1UL; i < seq_.station_ids_.size(); ++i) {  // arrivals
      auto ref =
          stub_strategy->close_nodes(seq_.station_ids_.at(i).str()).front();
      station_arrivals_.at(i).push_back(
          graph_.nodes_.emplace_back(std::make_unique<seq_node>(i, ref)).get());
    }

    auto const& collect_refs = [](auto const& vec) {
      return utl::to_vec(vec, [](auto const& node) { return node->ref_; });
    };

    for (auto i = 0UL; i < graph_.seq_size_ - 1; ++i) {  // between stations
      std::vector<seq_node*> from{station_departures_.at(i).back()};
      std::vector<seq_node*> to{station_arrivals_.at(i + 1).back()};
      make_edges(
          from, to, kInvalidPartTask,
          stub_strategy->find_routes(collect_refs(from), collect_refs(to)));
    }
    for (auto i = 1UL; i < graph_.seq_size_ - 1; ++i) {  // within station
      auto const from = station_arrivals_.at(i);
      auto const to = station_departures_.at(i);
      make_edges(
          from, to, kInvalidPartTask,
          stub_strategy->find_routes(collect_refs(from), collect_refs(to)));
    }
  }

  seq_graph finish() {
    size_t idx = 0;
    for (auto const& node : graph_.nodes_) {
      node->idx_ = idx++;

      if (node->station_idx_ == 0) {
        graph_.initials_.emplace_back(node->idx_);
      } else if (node->station_idx_ == (graph_.seq_size_ - 1)) {
        graph_.goals_.emplace_back(node->idx_);
      }
    }
    return std::move(graph_);
  }

  station_seq const& seq_;
  seq_graph graph_;

  std::vector<seq_node*> from_nodes_;
  std::vector<std::vector<seq_node*>> station_arrivals_, station_departures_;

  routing_strategy const* curr_strategy_ = nullptr;
  size_t curr_station_idx_ = std::numeric_limits<size_t>::max();
  bool curr_at_station_ = false;
};

}  // namespace motis::path
