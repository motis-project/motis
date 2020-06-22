#include "motis/path/prepare/osm_path.h"

namespace motis::path {

void osm_path::unique() {
  if (size() < 2) {
    return;
  }

  mcd::vector<geo::latlng> tmp_polyline{polyline_.front()};
  mcd::vector<int64_t> tmp_osm_node_ids{osm_node_ids_.front()};

  for (auto i = 1UL; i < size(); ++i) {
    if (osm_node_ids_[i] == kPathUnknownNodeId &&
        tmp_osm_node_ids.back() == kPathUnknownNodeId &&
        polyline_[i] == tmp_polyline.back()) {
      continue;  // unknown duplicate
    }

    if (osm_node_ids_[i] != kPathUnknownNodeId &&
        osm_node_ids_[i] == tmp_osm_node_ids.back()) {
      continue;  // known duplicate
    }

    tmp_polyline.push_back(polyline_[i]);
    tmp_osm_node_ids.push_back(osm_node_ids_[i]);
  }

  polyline_ = std::move(tmp_polyline);
  osm_node_ids_ = std::move(tmp_osm_node_ids);
}

void osm_path::remove_loops() {
  if (size() <= 2) {
    return;
  }

  struct elem {
    elem(int64_t osm_id, geo::latlng pos, size_t idx)
        : osm_id_(osm_id), pos_(pos), idx_(idx) {}

    int64_t osm_id_;
    geo::latlng pos_;
    size_t idx_;
  };

  std::vector<elem> work;
  work.reserve(size());
  while (true) {
    work.clear();
    for (auto i = 0UL; i < size(); ++i) {
      work.emplace_back(osm_node_ids_[i], polyline_[i], i);
    }

    std::sort(begin(work), end(work), [](auto const& lhs, auto const& rhs) {
      return std::tie(lhs.osm_id_, lhs.pos_, lhs.idx_) <
             std::tie(rhs.osm_id_, rhs.pos_, rhs.idx_);
    });
    auto it = std::adjacent_find(begin(work), end(work),
                                 [](elem const& lhs, elem const& rhs) {
                                   if (lhs.osm_id_ == -1 && rhs.osm_id_ == -1) {
                                     return lhs.pos_ == rhs.pos_;
                                   } else {
                                     return lhs.osm_id_ == rhs.osm_id_;
                                   }
                                 });

    if (it == end(work)) {
      break;
    }

    auto loop_begin = it->idx_;
    auto loop_end = (it + 1)->idx_;
    utl::verify(loop_begin < loop_end, "osm_path::remove_loops invalid loop");

    polyline_.erase(begin(polyline_) + loop_begin,  //
                    begin(polyline_) + loop_end);
    osm_node_ids_.erase(begin(osm_node_ids_) + loop_begin,
                        begin(osm_node_ids_) + loop_end);
  }
}

void osm_path::ensure_line() {
  if (size() > 1) {
    return;
  }

  auto coord = polyline_.back();
  polyline_.push_back(coord);

  auto node_id = osm_node_ids_.back();
  osm_node_ids_.push_back(node_id);
}

}  // namespace motis::path
