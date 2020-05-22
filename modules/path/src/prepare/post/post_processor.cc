#include "motis/path/prepare/post/post_processor.h"

#include <map>

#include "utl/erase_duplicates.h"
#include "utl/join.h"
#include "utl/parallel_for.h"
#include "utl/to_vec.h"

#include "motis/core/common/logging.h"

#include "motis/path/prepare/osm_path.h"
#include "motis/path/prepare/post/post_debug_info.h"
#include "motis/path/prepare/post/post_graph.h"

namespace ml = motis::logging;

namespace motis::path {

void mark_essentials(post_graph& graph) {
  ml::scoped_timer timer("post_processor|mark_essentials");
  utl::parallel_for_run(graph.nodes_.size(), [&](auto const i) {
    auto& node = graph.nodes_.at(i);

    std::set<color_t> pre_marked = node->essential_;
    std::map<std::vector<color_t>, size_t> degrees;

    auto const edge_cmp = [](auto const& a, auto const& b) {
      return a.other_->id_ < b.other_->id_;
    };
    std::sort(begin(node->out_), end(node->out_), edge_cmp);
    std::sort(begin(node->inc_), end(node->inc_), edge_cmp);
    utl::full_join(
        node->out_, node->inc_, edge_cmp,
        [&degrees](auto lb_out, auto ub_out, auto lb_inc, auto ub_inc) {
          utl::verify(std::distance(lb_out, ub_out) == 1, "duplicate edge (1)");
          utl::verify(std::distance(lb_inc, ub_inc) == 1, "duplicate edge (2)");

          // _all_ color between a pair of nodes count together
          std::vector<color_t> colors;
          utl::concat(colors, lb_out->colors_);
          utl::concat(colors, lb_inc->colors_);
          utl::erase_duplicates(colors);  // sort ...

          ++degrees[colors];
        },
        [&degrees](auto lb_out, auto ub_out) {
          utl::verify(std::distance(lb_out, ub_out) == 1, "duplicate edge (3)");
          ++degrees[lb_out->colors_];
        },
        [&degrees](auto lb_inc, auto ub_inc) {
          utl::verify(std::distance(lb_inc, ub_inc) == 1, "duplicate edge (4)");
          ++degrees[lb_inc->colors_];
        });

    for (auto const& [colors, degree] : degrees) {
      utl::verify(degree <= 2, "color cycle detected!");

      auto const has_pre_marked = std::any_of(
          begin(colors), end(colors),
          [&](auto c) { return pre_marked.find(c) != end(pre_marked); });

      if (degree != 2 || has_pre_marked) {
        node->essential_.insert(begin(colors), end(colors));
      }
    }
  });
}

void construct_atomic_path(post_graph& graph, post_graph_node* start_node,
                           post_graph_edge* start_edge) {
  std::vector<post_graph_node*> path;
  path.push_back(start_node);

  auto edge = start_edge;
  auto node = start_edge->other_;
  while (true) {
    path.push_back(node);
    if (node->is_essential_for(*edge)) {
      break;
    }

    edge = node->find_out_edge(start_edge->colors_);
    if (edge == nullptr) {
      break;
    }

    node = edge->other_;
  }

  utl::verify(path.size() >= 2, "construct_atomic_path: missing path");

  utl::verify(path.back()->is_essential_for(*start_edge),
              "non essential end point found");
  utl::verify(path.back() == node, "bad endpoint");

  auto rev_edge = node->find_edge_to(path[path.size() - 2]);
  if (rev_edge != nullptr && rev_edge->atomic_path_ != nullptr &&
      std::equal(begin(path), end(path),
                 rbegin(rev_edge->atomic_path_->path_))) {
    start_edge->atomic_path_ = rev_edge->atomic_path_;
    start_edge->atomic_path_forward_ = false;
  } else {
    graph.atomic_paths_.push_back(
        std::make_unique<atomic_path>(path, start_node, node));
    start_edge->atomic_path_ = graph.atomic_paths_.back().get();
    start_edge->atomic_path_forward_ = true;
  }
}

void find_atomic_paths(post_graph& graph) {
  ml::scoped_timer timer("post_processor|find_atomic_paths");

  for (auto const& node : graph.nodes_) {
    for (auto& edge : node->out_) {
      if (edge.atomic_path_ != nullptr) {
        continue;
      }

      if (!node->is_essential_for(edge)) {
        continue;
      }

      construct_atomic_path(graph, node.get(), &edge);
    }
  }
}

void post_process(post_graph& graph) {
  mark_essentials(graph);
  find_atomic_paths(graph);

  LOG(ml::info) << "postprocessed " << graph.atomic_paths_.size() << " paths";
}

}  // namespace motis::path
