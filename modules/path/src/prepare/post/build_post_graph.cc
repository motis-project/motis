#include "motis/path/prepare/post/build_post_graph.h"

#include <limits>

#include "utl/erase_duplicates.h"
#include "utl/parallel_for.h"
#include "utl/to_vec.h"

#include "motis/core/common/logging.h"

#include "motis/path/prepare/resolve/thread_pool.h"

namespace ml = motis::logging;

namespace motis::path {

struct post_graph_builder {
  explicit post_graph_builder(std::vector<resolved_station_seq> seq)
      : graph_{std::move(seq)}, color_(0) {}

  void make_nodes() {
    ml::scoped_timer t{"build_post_graph|make_nodes"};

    for (auto const& seq : graph_.originals_) {
      for (auto const& path : seq.paths_) {
        for (auto i = 0UL; i < path.size(); ++i) {
          node_ids_.emplace_back(path.osm_node_ids_[i], path.polyline_[i]);
        }
      }
    }
    LOG(ml::info) << "pre " << node_ids_.size();
    utl::erase_duplicates(node_ids_);  // post condition: sorted
    LOG(ml::info) << "post " << node_ids_.size();

    node_mutex_ = std::vector<std::mutex>(node_ids_.size());

    graph_.nodes_ = utl::to_vec(node_ids_, [](auto const& id) {
      return std::make_unique<post_graph_node>(id);
    });
  }

  void make_edges() {
    ml::scoped_timer t{"build_post_graph|make_edges"};

    thread_local std::vector<color_t> node_max_colors;  // node -> max color
    thread_pool tp{
        [&] { node_max_colors.resize(node_ids_.size(), kInvalidColor); },
        [&] { node_max_colors = std::vector<color_t>(); }};

    graph_.segment_ids_.resize(graph_.originals_.size());
    tp.execute(graph_.originals_.size(), [this](auto const i) {
      graph_.segment_ids_.at(i) =
          append_seq(graph_.originals_.at(i), node_max_colors);
    });

    tp.execute(graph_.nodes_.size(), [this](auto const i) {
      auto& node = graph_.nodes_.at(i);
      for (auto& out_edge : node->out_) {
        utl::erase_duplicates(out_edge.colors_);
      }
      for (auto& inc_edge : node->inc_) {
        utl::erase_duplicates(inc_edge.colors_);
      }
    });
  }

  std::vector<post_segment_id> append_seq(
      resolved_station_seq const& seq, std::vector<color_t>& node_max_colors) {
    auto color = next_seq_color();

    std::vector<post_segment_id> segment_ids;
    for (auto const& path : seq.paths_) {
      post_graph_node* prev = nullptr;
      size_t prev_idx = 0;

      for (auto i = 0UL; i < path.size(); ++i) {
        auto const curr_idx =
            get_node_idx({path.osm_node_ids_[i], path.polyline_[i]});
        auto* curr = graph_.nodes_.at(curr_idx).get();

        if (prev == nullptr) {
          segment_ids.emplace_back(curr, color);
        } else if (curr != prev) {
          if (node_max_colors[curr_idx] == color) {
            ++color;  // prevent color cycles
          }

          std::scoped_lock lock(node_mutex_[curr_idx], node_mutex_[prev_idx]);
          add_color(curr->inc_, prev, color);
          add_color(prev->out_, curr, color);
        }

        node_max_colors[curr_idx] = color;
        node_max_colors[prev_idx] = color;

        prev = curr;
        prev_idx = curr_idx;
      }

      segment_ids.back().max_color_ = color;
      ++color;
    }
    return segment_ids;
  }

  color_t next_seq_color() { return (color_++) << 16U; };

  size_t get_node_idx(post_node_id const& id) {
    auto const it = std::lower_bound(begin(node_ids_), end(node_ids_), id);
    utl::verify(it != end(node_ids_) && *it == id, "post node not found");
    return std::distance(begin(node_ids_), it);
  }

  static void add_color(std::vector<post_graph_edge>& edges,
                        post_graph_node* node, color_t const color) {
    auto it = std::find_if(begin(edges), end(edges), [&](auto const& edge) {
      return edge.other_ == node;
    });

    if (it != end(edges)) {
      it->colors_.push_back(color);
    } else {
      edges.emplace_back(node, std::vector<color_t>{color});
    }
  }

  post_graph graph_;

  std::vector<post_node_id> node_ids_;  // node -> node_id
  std::vector<std::mutex> node_mutex_;  // node -> mutex

  std::atomic<color_t> color_;
};

void check_out_colors(post_graph const& graph) {
  utl::parallel_for(graph.nodes_, [](auto const& n) {
    std::vector<color_t> colors;

    for (auto const& edge : n->out_) {
      utl::concat(colors, edge.colors_);
    }

    auto size_before = colors.size();
    utl::erase_duplicates(colors);
    auto size_after = colors.size();
    utl::verify(size_before == size_after, "color size mismatch!");
  });
}

post_graph build_post_graph(std::vector<resolved_station_seq> seq) {
  ml::scoped_timer t{"build_post_graph"};

  post_graph_builder builder{std::move(seq)};
  builder.make_nodes();
  builder.make_edges();

  check_out_colors(builder.graph_);

  return std::move(builder.graph_);
}

}  // namespace motis::path
