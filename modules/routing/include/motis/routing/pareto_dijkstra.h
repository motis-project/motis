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
#define ROUTING_DEBUG_ENABLED
#ifdef ROUTING_DEBUG_ENABLED
  struct label_dbg {
    schedule const& sched_;
    Label const& l_;
  };

  struct edge_dbg {
    schedule const& sched_;
    edge const& e_;
  };

  friend std::ostream& operator<<(std::ostream& out, label_dbg const& dbg_l) {
    auto const [sched, l] = dbg_l;

    out << "{LABEL start=" << format_time(l.start_)
        << ", now=" << format_time(l.now_) << " with (";

    if constexpr (std::is_base_of_v<travel_time, Label>) {
      out << "tt=" << format_time(l.travel_time_) << ", ";
    }

    if constexpr (std::is_base_of_v<transfers, Label>) {
      out << "ic=" << static_cast<int>(l.transfers_) << ", ";
    }

    if constexpr (std::is_base_of_v<travel_time, Label>) {
      out << "tt_lb=" << format_time(l.travel_time_lb_) << ", ";
    }

    if constexpr (std::is_base_of_v<transfers, Label>) {
      out << "ic_lb=" << static_cast<int>(l.transfers_lb_);
    }

    auto const node = l.edge_->to_;
    out << ") AT " << node->type_str() << "=" << node->id_
        << ", station=" << sched.stations_.at(node->get_station()->id_)->name_
        << "}";

    return out;
  }

  friend std::ostream& operator<<(std::ostream& out, edge_dbg const& dbg_e) {
    auto const& [sched, e] = dbg_e;
    return out << "{" << e.type_str() << " from=" << e.from_->id_ << "["
               << sched.stations_.at(e.from_->get_station()->id_)->name_
               << "], to=" << e.to_->id_ << " ["
               << sched.stations_.at(e.to_->get_station()->id_)->name_ << "]"
               << "}";
  }

  template <typename FmtStr, typename... FmtArgs>
  void dbg(FmtStr fmt_str, FmtArgs&&... args) {
    fmt::print(std::cerr, std::forward<FmtStr>(fmt_str),
               std::forward<FmtArgs&&>(args)...);
    std::cerr << "\n";
//    logging::l(logging::log_level::debug, std::forward<FmtStr>(fmt_str),
//               std::forward<FmtArgs&&>(args)...);
  }
#else
  template <typename FmtStr, typename FmtArgs>
  void dbg(FmtStr, FmtArgs&&...) {}
#endif

  struct compare_labels {
    bool operator()(Label const* a, Label const* b) const {
      return a->operator<(*b);
    }
  };

  struct get_bucket {
    std::size_t operator()(Label const* l) const { return l->get_bucket(); }
  };

  pareto_dijkstra(
      schedule const& sched, boost::container::vector<bool> const& is_goal,
      mcd::hash_map<node const*, std::vector<edge>> additional_edges,
      LowerBounds& lower_bounds, mem_manager& label_store)
      : sched_{sched},
        is_goal_{is_goal},
        station_node_count_{sched.station_nodes_.size()},
        node_labels_{*label_store.get_node_labels<Label>(sched.next_node_id_)},
        additional_edges_{std::move(additional_edges)},
        lower_bounds_{lower_bounds},
        label_store_{label_store},
        max_labels_{1024ULL * 1024 * 128} {}

  void add_start_labels(std::vector<Label*> const& start_labels) {
    for (auto const& l : start_labels) {
      if (!l->is_filtered()) {
        dbg("START_LABEL: {}", label_dbg{sched_, *l});
        node_labels_[l->get_node()->id_].emplace_back(l);
        queue_.push(l);
      } else {
        dbg("FILTERED_START_LABEL: {}", label_dbg{sched_, *l});
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

      dbg("extract label {}", label_dbg{sched_, *label});

      // is label already made obsolete
      if (label->dominated_) {
        dbg("\tdominated");
        label_store_.release(label);
        stats_.labels_dominated_by_later_labels_++;
        continue;
      }

      if (dominated_by_results(label)) {
        dbg("\tdominated by results");
        stats_.labels_dominated_by_results_++;
        continue;
      }

      if (label->get_node()->id_ < station_node_count_ &&
          is_goal_[label->get_node()->id_]) {
        dbg("\tat goal");
        continue;
      }

      dbg("\texpanding:");
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

  void create_new_label(Label* l, edge const& edge) {
    Label blank{};
    auto const status = l->create_label(
        blank, edge, lower_bounds_,
        (Dir == search_dir::FWD && edge.type() == edge::EXIT_EDGE &&
         is_goal_[edge.get_source<Dir>()->get_station()->id_]) ||
            (Dir == search_dir::BWD && edge.type() == edge::ENTER_EDGE &&
             is_goal_[edge.get_source<Dir>()->get_station()->id_]));
    if (status != create_label_result::CREATED) {
      dbg("\t\t{} -> {} -- no label created ({})", label_dbg{sched_, *l},
          edge_dbg{sched_, edge},
          create_label_result_str[static_cast<
              std::underlying_type_t<create_label_result>>(status)]);
      return;
    }

    auto new_label = label_store_.create<Label>(blank);
    ++stats_.labels_created_;
    dbg("\t\t{} -> {} -- label created", label_dbg{sched_, *l},
        edge_dbg{sched_, edge});

    if (edge.get_destination<Dir>()->id_ < station_node_count_ &&
        is_goal_[edge.get_destination<Dir>()->id_]) {
      dbg("\t\t\tRESULT: {}", label_dbg{sched_, *new_label});
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
      dbg("\t\t\tDOMINATED_BY_RESULTS: {}", label_dbg{sched_, *new_label});
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
        dbg("\t\t\tRESULT_DOMINATED_BY_EXISTING:");
        dbg("\t\t\t\t     NEW: {}", label_dbg{sched_, *terminal_label});
        dbg("\t\t\t\tEXISTING: {}", label_dbg{sched_, *o});
        return false;
      } else {
        ++it;
      }
    }
    dbg("\t\t\tNEW_RESULT: {}", label_dbg{sched_, *terminal_label});
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
        dbg("\t\t\tEXISTING_DOMINATED_BY_NEW:");
        dbg("\t\t\t\t     NEW: {}", label_dbg{sched_, *new_label});
        dbg("\t\t\t\tEXISTING: {}", label_dbg{sched_, *o});

        it = dest_labels.erase(it);
        o->dominated_ = true;
      } else {
        ++it;
      }
    }

    // it is very important for the performance to push front here
    // because earlier labels tend not to dominate later ones (not comparable)
    dbg("\t\t\tNEW_LABEL_AT_NODE: {}", label_dbg{sched_, *new_label});
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

  schedule const& sched_;
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
