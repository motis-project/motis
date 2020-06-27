#include "motis/path/prepare/post/build_post_graph.h"

#include <limits>

#include "utl/erase_duplicates.h"
#include "utl/parallel_for.h"
#include "utl/thread_pool.h"
#include "utl/to_vec.h"

#include "motis/core/common/logging.h"

namespace ml = motis::logging;

namespace motis::path {

struct post_graph_builder {
  explicit post_graph_builder(mcd::unique_ptr<mcd::vector<station_seq>> seq)
      : graph_{std::move(seq)}, color_(0) {}

  void make_nodes() {
    ml::scoped_timer t{"build_post_graph|make_nodes"};

    for (auto const& seq : *graph_.originals_) {
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

    // node -> max color
    thread_local std::unique_ptr<std::vector<color_t>> node_max_colors;
    utl::thread_pool tp{
        [&] {
          node_max_colors = std::make_unique<std::vector<color_t>>();
          node_max_colors->resize(node_ids_.size(), kInvalidColor);
        },
        [&] { node_max_colors.reset(); }};

    graph_.segment_ids_.resize(graph_.originals_->size());
    tp.execute(graph_.originals_->size(), [this](auto const i) {
      graph_.segment_ids_.at(i) =
          append_seq(graph_.originals_->at(i), *node_max_colors);
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
      station_seq const& seq, std::vector<color_t>& node_max_colors) {
    auto color = next_seq_color();

    struct edge_color {
      size_t from_idx_, to_idx_;
      color_t color_;
    };
    std::vector<edge_color> edges;

    std::vector<post_segment_id> segment_ids;
    for (auto const& path : seq.paths_) {
      // path empty or a self loop
      if (path.size() == 0 ||
          (get_node_idx({path.osm_node_ids_.front(), path.polyline_.front()}) ==
           get_node_idx({path.osm_node_ids_.back(), path.polyline_.back()}))) {
        segment_ids.emplace_back();
        continue;
      }

      edges.clear();
      auto const base_color = color;

      size_t prev_idx = 0;
      for (auto i = 0ULL; i < path.size(); ++i) {
        auto const curr_idx =
            get_node_idx({path.osm_node_ids_[i], path.polyline_[i]});

        if (i == 0ULL) {  // first node
          node_max_colors[curr_idx] = color;
          prev_idx = curr_idx;
          continue;
        }

        if (curr_idx == prev_idx) {  // no progress
          continue;
        }

        if (node_max_colors[curr_idx] == color) {  // self-loop detected
          // increment color for edges inside the loop
          ++color;

          // re-color all edges inside the loop
          for (auto it = std::rbegin(edges); it != std::rend(edges); ++it) {
            it->color_ = color;
            node_max_colors[it->to_idx_] = color;

            if (it->from_idx_ == curr_idx) {
              break;
            }
          }

          // add loop closing edge
          edges.push_back({prev_idx, curr_idx, color});
          node_max_colors[curr_idx] = color;

          // increment color for edges after the loop
          ++color;
        } else {  // no loop
          edges.push_back({prev_idx, curr_idx, color});
          node_max_colors[curr_idx] = color;
          node_max_colors[prev_idx] = color;
        }

        prev_idx = curr_idx;
      }

      if (edges.empty()) {
        segment_ids.emplace_back();
        continue;
      }

      utl::verify(edges.back().color_ <
                      base_color + std::numeric_limits<uint16_t>::max(),
                  "append_seq: too many color increments!");

      auto color_offset = 0;
      for (auto i = 0ULL; i < edges.size(); ++i) {
        auto const increment_color =
            i != 0 && edges[i].color_ != edges[i - 1].color_;

        if (increment_color) {
          ++color_offset;  // ensure two adjacent edges differ by max 1
        }

        auto const current_color = base_color + color_offset;

        auto const& edge = edges[i];
        std::scoped_lock lock(node_mutex_[edge.from_idx_],
                              node_mutex_[edge.to_idx_]);

        auto* from = graph_.nodes_.at(edge.from_idx_).get();
        auto* to = graph_.nodes_.at(edge.to_idx_).get();

        add_color(from->out_, to, current_color);
        add_color(to->inc_, from, current_color);

        // pre-mark essentials as long as the cycle start is obvious
        if (increment_color) {
          from->essential_.insert(current_color);  // current color
          from->essential_.insert(current_color - 1);  // prev color
        }
      }

      auto const final_color = base_color + color_offset;
      segment_ids.emplace_back(graph_.nodes_.at(edges.front().from_idx_).get(),
                               graph_.nodes_.at(edges.back().to_idx_).get(),
                               base_color, final_color);

      // increment color for the next segment
      color = final_color + 1;
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

post_graph build_post_graph(mcd::unique_ptr<mcd::vector<station_seq>> seq) {
  ml::scoped_timer t{"build_post_graph"};

  post_graph_builder builder{std::move(seq)};
  builder.make_nodes();
  builder.make_edges();

  check_out_colors(builder.graph_);

  return std::move(builder.graph_);
}

}  // namespace motis::path
