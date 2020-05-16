#include "motis/path/prepare/post/post_processor.h"

#include <map>

#include "utl/erase_duplicates.h"
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
    std::map<std::vector<color_t>, size_t> degrees;
    for (auto const& inc : node->inc_) {
      ++degrees[inc.colors_];
    }
    for (auto const& out : node->out_) {
      ++degrees[out.colors_];
    }

    for (auto const& pair : degrees) {
      if (pair.second > 2) {
        // TODO(sebastian) check these!!

        // std::clog << "color cycle: " << pair.second << std::endl;
        // for (auto const color : pair.first) {
        //   print_post_colors(graph, color);
        // }
      }
      // verify(pair.second <= 2, "color cycle detected!");

      if (pair.second != 2) {
        node->essential_.insert(begin(pair.first), end(pair.first));
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

    edge = node->find_out_edge(start_edge->colors_);
    if (edge == nullptr) {
      break;
    }

    node = edge->other_;
    if (node->is_essential_for(*edge)) {
      path.push_back(node);
      break;
    }
  }

  if (path.size() == 2) {
    graph.atomic_pairs_.emplace_back(std::min(path[0], path[1]),
                                     std::max(path[0], path[1]));
    return;
  } else if (path.size() < 2) {
    return;  // cant optimize this
  }

  auto rev_edge = node->find_edge_to(path[path.size() - 2]);
  if (rev_edge != nullptr && rev_edge->atomic_path_ != nullptr &&
      std::equal(begin(path), end(path),
                 rbegin(rev_edge->atomic_path_->path_))) {
    start_edge->atomic_path_ = rev_edge->atomic_path_;
    start_edge->atomic_path_forward_ = false;
  } else {
    graph.atomic_paths_.push_back(
        std::make_unique<atomic_path>(path, start_node, node));
    auto* ap = graph.atomic_paths_.back().get();

    start_edge->atomic_path_ = ap;
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

void simplify_atomic_paths(post_graph& graph) {
  ml::scoped_timer timer("post_processor|simplify_atomic_paths");

  // XX parallel for
  for (auto& ap : graph.atomic_paths_) {
    auto polyline =
        utl::to_vec(ap->path_, [](auto const& node) { return node->id_.pos_; });
    ap->mask_ = make_simplify_mask(polyline, 8);
  }
}

void post_process(post_graph& graph) {
  // print_post_graph(graph);

  mark_essentials(graph);
  find_atomic_paths(graph);
  simplify_atomic_paths(graph);

  utl::erase_duplicates(graph.atomic_pairs_);

  LOG(ml::info) << "postprocessed " << graph.atomic_paths_.size()
                << " paths and " << graph.atomic_pairs_.size() << " pairs";
}

}  // namespace motis::path
