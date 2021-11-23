#pragma once

#include <queue>

#include "boost/container/vector.hpp"
#include "utl/erase_if.h"

#include "motis/hash_map.h"

#include "motis/core/common/dial.h"

#include "motis/routing/mem_manager.h"
#include "motis/routing/statistics.h"

namespace motis::routing {

const bool FORWARDING = true;

template <search_dir Dir, typename Label, typename LowerBounds>
struct pareto_dijkstra {
  struct compare_labels {
    bool operator()(Label const* a, Label const* b) const {
      return a->operator<(*b);
    }
  };

  struct get_bucket {
    std::size_t operator()(Label const* l) const { return l->get_bucket(); }
  };

  pareto_dijkstra(
      int node_count, unsigned int station_node_count,
      boost::container::vector<bool> const& is_goal,
      mcd::hash_map<node const*, std::vector<edge>> additional_edges,
      LowerBounds& lower_bounds, mem_manager& label_store)
      : is_goal_(is_goal),
        station_node_count_(station_node_count),
        node_labels_(*label_store.get_node_labels<Label>(node_count)),
        additional_edges_(std::move(additional_edges)),
        lower_bounds_(lower_bounds),
        label_store_(label_store),
        max_labels_(1024 * 1024 * 128) {}

  void add_start_labels(std::vector<Label*> const& start_labels) {
    for (auto const& l : start_labels) {
      if (!l->is_filtered()) {
        node_labels_[l->get_node()->id_].emplace_back(l);
        queue_.push(l);
      }
    }
  }

  void search() {
    stats_.start_label_count_ = queue_.size();
    stats_.labels_created_ = label_store_.allocations();

    while (!queue_.empty() || !equals_.empty()) {
      if ((stats_.labels_created_ > (max_labels_ / 2) && results_.empty()) ||
          stats_.labels_created_ > max_labels_) {
        stats_.max_label_quit_ = true;
        filter_results();
        return;
      }

      // get best label
      Label* label = nullptr;
      if (!equals_.empty()) {
        label = equals_.back();
        equals_.pop_back();
        stats_.labels_equals_popped_++;
      } else {
        label = queue_.top();
        stats_.priority_queue_max_size_ =
            std::max(stats_.priority_queue_max_size_,
                     static_cast<uint64_t>(queue_.size()));
        queue_.pop();
        stats_.labels_popped_++;
        stats_.labels_popped_after_last_result_++;
      }

      // is label already made obsolete
      if (label->dominated_) {
        label_store_.release(label);
        stats_.labels_dominated_by_later_labels_++;
        continue;
      }

      if (dominated_by_results(label)) {
        stats_.labels_dominated_by_results_++;
        continue;
      }

      if (label->get_node()->id_ < station_node_count_ &&
          is_goal_[label->get_node()->id_]) {
        continue;
      }

      auto it = additional_edges_.find(label->get_node());
      if (it != std::end(additional_edges_)) {
        for (auto const& additional_edge : it->second) {
          create_new_label(label, additional_edge);
        }
      }

      if (Dir == search_dir::FWD) {
        for (auto const& edge : label->get_node()->edges_) {
          create_new_label(label, edge);
        }
      } else {
        for (auto const& edge : label->get_node()->incoming_edges_) {
          create_new_label(label, *edge);
        }
      }
    }

    filter_results();
  }

  statistics get_statistics() const { return stats_; };

  std::vector<Label*> const& get_results() { return results_; }

private:
  void create_new_label(Label* l, edge const& edge) {
    Label blank{};
    bool created = l->create_label(
        blank, edge, lower_bounds_,
        (Dir == search_dir::FWD && edge.type() == edge::EXIT_EDGE &&
         is_goal_[edge.get_source<Dir>()->get_station()->id_]) ||
            (Dir == search_dir::BWD && edge.type() == edge::ENTER_EDGE &&
             is_goal_[edge.get_source<Dir>()->get_station()->id_]));
    if (!created) {
      return;
    }

    auto new_label = label_store_.create<Label>(blank);
    ++stats_.labels_created_;

    if (edge.get_destination<Dir>()->id_ < station_node_count_ &&
        is_goal_[edge.get_destination<Dir>()->id_]) {
      add_result(new_label);
      if (stats_.labels_popped_until_first_result_ == 0) {
        stats_.labels_popped_until_first_result_ = stats_.labels_popped_;
      }
      return;
    }

    // if the label is not dominated by a former one for the same node...
    //...add it to the queue
    if (!dominated_by_results(new_label)) {
      if (add_label_to_node(new_label, edge.get_destination<Dir>())) {
        // if the new_label is as good as label we don't have to push it into
        // the queue
        if (!FORWARDING || l < new_label) {
          queue_.push(new_label);
        } else {
          equals_.push_back(new_label);
        }
      } else {
        label_store_.release(new_label);
        stats_.labels_dominated_by_former_labels_++;
      }
    } else {
      label_store_.release(new_label);
      stats_.labels_dominated_by_results_++;
    }
  }

  bool add_result(Label* terminal_label) {
    for (auto it = results_.begin(); it != results_.end();) {
      Label* o = *it;
      if (terminal_label->dominates(*o)) {
        label_store_.release(o);
        it = results_.erase(it);
      } else if (o->dominates(*terminal_label)) {
        return false;
      } else {
        ++it;
      }
    }
    results_.push_back(terminal_label);
    stats_.labels_popped_after_last_result_ = 0;
    return true;
  }

  bool add_label_to_node(Label* new_label, node const* dest) {
    auto& dest_labels = node_labels_[dest->id_];
    for (auto it = dest_labels.begin(); it != dest_labels.end();) {
      Label* o = *it;
      if (o->dominates(*new_label)) {
        return false;
      }

      if (new_label->dominates(*o)) {
        it = dest_labels.erase(it);
        o->dominated_ = true;
      } else {
        ++it;
      }
    }

    // it is very important for the performance to push front here
    // because earlier labels tend not to dominate later ones (not comparable)
    dest_labels.insert(std::begin(dest_labels), new_label);
    return true;
  }

  bool dominated_by_results(Label* label) {
    return std::any_of(begin(results_), end(results_), [&](auto&& result) {
      return result->dominates(*label);
    });
  }

  void filter_results() {
    if (!Label::is_post_search_dominance_enabled()) {
      return;
    }
    bool restart = false;
    for (auto it = std::begin(results_); it != std::end(results_);
         it = restart ? std::begin(results_) : std::next(it)) {
      restart = false;
      std::size_t size_before = results_.size();
      utl::erase_if(results_, [it](Label const* l) {
        return l != (*it) && (*it)->dominates_post_search(*l);
      });
      if (results_.size() != size_before) {
        restart = true;
      }
    }
  }

  boost::container::vector<bool> const& is_goal_;
  unsigned int station_node_count_;
  std::vector<std::vector<Label*>>& node_labels_;
  dial<Label*, Label::MAX_BUCKET, get_bucket> queue_;
  std::vector<Label*> equals_;
  mcd::hash_map<node const*, std::vector<edge>> additional_edges_;
  std::vector<Label*> results_;
  LowerBounds& lower_bounds_;
  mem_manager& label_store_;
  statistics stats_;
  std::size_t max_labels_;
};

}  // namespace motis::routing
