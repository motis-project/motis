#pragma once

#include <memory>
#include <set>

#include "cista/hashing.h"

#include "geo/box.h"
#include "geo/polyline.h"

#include "motis/core/common/hash_helper.h"

#include "motis/path/prepare/osm_path.h"
#include "motis/path/prepare/schedule/station_sequences.h"

namespace motis::path {

using color_t = size_t;
constexpr auto const kInvalidColor = std::numeric_limits<color_t>::max();

struct post_graph_node;

struct post_segment_id {
  post_segment_id()
      : front_{nullptr},
        back_{nullptr},
        color_{kInvalidColor},
        max_color_{kInvalidColor} {}

  post_segment_id(post_graph_node* front, post_graph_node* back,  //
                  color_t color, color_t max_color)
      : front_{front}, back_{back}, color_{color}, max_color_{max_color} {}

  [[nodiscard]] bool valid() const { return front_ != nullptr; }

  post_graph_node *front_, *back_;
  color_t color_, max_color_;
};

struct post_node_id {
  post_node_id() = default;
  post_node_id(int64_t osm_id, geo::latlng pos) : osm_id_(osm_id), pos_(pos) {}

  friend bool operator<(post_node_id const& lhs, post_node_id const& rhs) {
    if (lhs.osm_id_ == -1 && rhs.osm_id_ == -1) {
      return lhs.pos_ < rhs.pos_;
    } else {
      return lhs.osm_id_ < rhs.osm_id_;
    }
  }

  friend bool operator==(post_node_id const& lhs, post_node_id const& rhs) {
    if (lhs.osm_id_ == -1 && rhs.osm_id_ == -1) {
      return lhs.pos_ == rhs.pos_;
    } else {
      return lhs.osm_id_ == rhs.osm_id_;
    }
  }

  struct hash {
    std::size_t operator()(motis::path::post_node_id const& id) const {
      return id.osm_id_ == -1
                 ? cista::build_hash(id.osm_id_)
                 : cista::build_hash(id.osm_id_, id.pos_.lat_, id.pos_.lng_);
    }
  };

  int64_t osm_id_ = 0;
  geo::latlng pos_;
};

struct atomic_path {
  atomic_path(std::vector<post_graph_node*> path, post_graph_node* from,
              post_graph_node* to)
      : path_(std::move(path)), from_(from), to_(to) {}

  std::vector<post_graph_node*> path_;

  post_graph_node* from_;
  post_graph_node* to_;

  uint64_t id_{0}, hint_{0};
  geo::box box_{};
};

struct post_graph_edge {
  post_graph_edge(post_graph_node* other, std::vector<color_t> colors)
      : other_(other),
        colors_(std::move(colors)),
        atomic_path_(nullptr),
        atomic_path_forward_(true) {}

  post_graph_node* other_;
  std::vector<color_t> colors_;

  atomic_path* atomic_path_ = nullptr;
  bool atomic_path_forward_ = true;
};

struct post_graph_node {
  explicit post_graph_node(post_node_id id) : id_{id} {}

  bool is_essential_for(post_graph_edge const& edge) const {
    return essential_.find(*begin(edge.colors_)) != end(essential_);
  }

  post_graph_edge* find_out_edge(std::vector<color_t> const& colors) {
    auto it = std::find_if(begin(out_), end(out_), [colors](auto const& edge) {
      return edge.colors_ == colors;
    });
    return it == end(out_) ? nullptr : &*it;
  }

  post_graph_edge* find_out_edge(color_t const& color) {
    auto it = std::find_if(begin(out_), end(out_), [color](auto const& edge) {
      return std::find(begin(edge.colors_), end(edge.colors_), color) !=
             end(edge.colors_);
    });
    return it == end(out_) ? nullptr : &*it;
  }

  post_graph_edge* find_edge_to(post_graph_node const* node) {
    auto it = std::find_if(begin(out_), end(out_), [node](auto const& edge) {
      return edge.other_ == node;
    });
    return it == end(out_) ? nullptr : &*it;
  }

  post_node_id id_;
  std::vector<post_graph_edge> out_, inc_;

  std::set<color_t> essential_;
};

struct post_graph {
  post_graph() = default;
  explicit post_graph(mcd::unique_ptr<mcd::vector<station_seq>> originals)
      : originals_{std::move(originals)} {}

  ~post_graph() = default;

  post_graph(post_graph const&) noexcept = delete;  // NOLINT
  post_graph& operator=(post_graph const&) noexcept = delete;  // NOLINT

  post_graph(post_graph&&) noexcept = default;  // NOLINT
  post_graph& operator=(post_graph&&) noexcept = default;  // NOLINT

  // parallel vectors!
  mcd::unique_ptr<mcd::vector<station_seq>> originals_;
  std::vector<std::vector<post_segment_id>> segment_ids_;

  // mem
  std::vector<std::unique_ptr<post_graph_node>> nodes_;
  std::vector<std::unique_ptr<atomic_path>> atomic_paths_;
};

}  // namespace motis::path
