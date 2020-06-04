#pragma once

#include "utl/thread_pool.h"

#include "motis/path/prepare/osm_util.h"

namespace motis::path {

struct raw_node {
  raw_node(int64_t id, int64_t way)
      : id_(id), resolved_(false), in_ways_({way}) {}

  int64_t id_;
  bool resolved_;
  geo::latlng pos_;
  std::vector<int64_t> in_ways_;
};

struct raw_way {
  explicit raw_way(int64_t id) : id_(id), oneway_(false), resolved_(false) {}

  raw_way(int64_t id, bool oneway, std::vector<raw_node*> nodes)
      : id_(id), oneway_(oneway), resolved_(false), nodes_(std::move(nodes)) {}

  int64_t id_;
  bool oneway_;
  bool resolved_;
  std::vector<raw_node*> nodes_;
};

template <typename Map>
void resolve_node_locations(std::string const& osm_file, Map& pending_nodes) {
  foreach_osm_node(osm_file, [&](auto const& node) {
    auto const it = pending_nodes.find(node.id());
    if (it != end(pending_nodes)) {
      it->second->resolved_ = true;
      it->second->pos_ = {node.location().lat(), node.location().lon()};
    }
  });
}

template <typename Map>
void resolve_node_locations(std::string const& osm_file, Map& pending_nodes,
                            utl::thread_pool& tp) {
  std::vector<std::pair<int64_t, geo::latlng>> buffer;
  auto const flush_buffer = [&] {
    tp.execute(buffer.size(), [&](auto i) {
      auto const& [node_id, pos] = buffer[i];
      auto const it = pending_nodes.find(node_id);
      if (it != end(pending_nodes)) {
        it->second->resolved_ = true;
        it->second->pos_ = pos;
      }
    });
    buffer.clear();
  };

  foreach_osm_node(osm_file, [&](auto const& node) {
    buffer.emplace_back(
        node.id(), geo::latlng{node.location().lat(), node.location().lon()});

    if (buffer.size() > 1e7) {
      flush_buffer();
    }
  });
  flush_buffer();
}

inline std::vector<osm_way> make_osm_ways(
    std::vector<raw_way> const& raw_ways) {
  std::vector<osm_way> osm_ways;
  auto const extract_way = [&osm_ways](auto const raw, auto const from,
                                       auto const to) {
    if (from == to) {
      return;
    }

    osm_path path{static_cast<size_t>(std::distance(from, to))};
    for (auto it = from; it != std::next(to); ++it) {
      path.polyline_.emplace_back((*it)->pos_);
      path.osm_node_ids_.emplace_back((*it)->id_);
    }

    osm_ways.emplace_back((*from)->id_, (*to)->id_, raw.id_, std::move(path),
                          raw.oneway_);
  };

  for (auto const& way : raw_ways) {
    if (way.nodes_.size() < 2) {
      continue;
    }

    auto from = begin(way.nodes_);
    while (true) {
      auto to = std::find_if(
          std::next(from), end(way.nodes_),
          [](auto const& node) { return node->in_ways_.size() > 1; });

      if (to == end(way.nodes_)) {
        break;
      }

      extract_way(way, from, to);
      from = to;
    }
    if (std::distance(from, end(way.nodes_)) >= 2) {
      extract_way(way, from, std::next(end(way.nodes_), -1));
    }
  }

  return osm_ways;
}

}  // namespace motis::path
