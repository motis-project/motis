#include "motis/path/prepare/post/post_serializer.h"

#include "utl/concat.h"
#include "utl/erase_duplicates.h"
#include "utl/parallel_for.h"
#include "utl/to_vec.h"

#include "motis/core/common/logging.h"

#include "motis/path/prepare/db_builder.h"
#include "motis/path/prepare/osm_path.h"
#include "motis/path/prepare/post/post_graph.h"

namespace ml = motis::logging;

namespace motis::path {

struct segment_builder {
  void append_node(post_graph_node const* node) {
    seg_.polyline_.push_back(node->id_.pos_);
    seg_.osm_ids_.push_back(node->id_.osm_id_);

    for (auto& mlvl : seg_.mask_) {
      mlvl.push_back(true);
    }
  }

  void append_atomic_path(atomic_path const* p, bool const forward) {
    if (forward) {
      for (auto i = 0UL; i < p->path_.size() - 1; ++i) {
        seg_.polyline_.push_back(p->path_[i]->id_.pos_);
        seg_.osm_ids_.push_back(p->path_[i]->id_.osm_id_);

        for (auto j = 0UL; j < p->mask_.size(); ++j) {
          seg_.mask_[j].push_back(p->mask_[j][i]);
        }
      }
    } else {
      for (auto i = p->path_.size() - 1; i > 0; --i) {
        seg_.polyline_.push_back(p->path_[i]->id_.pos_);
        seg_.osm_ids_.push_back(p->path_[i]->id_.osm_id_);

        for (auto j = 0UL; j < p->mask_.size(); ++j) {
          seg_.mask_[j].push_back(p->mask_[j][i]);
        }
      }
    }
  }

  processed_segment seg_;
};

processed_segment reconstruct_segment(post_graph const&,
                                      post_segment_id const& segment_id) {
  segment_builder sb;

  auto node = segment_id.start_;
  auto color = segment_id.color_;
  while (true) {
    post_graph_edge* edge = nullptr;
    if (color < segment_id.max_color_) {
      edge = node->find_out_edge(color + 1);
    }

    if (edge != nullptr) {
      ++color;
    } else {
      edge = node->find_out_edge(color);
    }

    if (edge == nullptr) {
      break;
    }

    if (edge->atomic_path_ == nullptr) {
      sb.append_node(node);
      node = edge->other_;
    } else {
      sb.append_atomic_path(edge->atomic_path_, edge->atomic_path_forward_);
      node = edge->atomic_path_forward_ ? edge->atomic_path_->to_
                                        : edge->atomic_path_->from_;
    }
  }

  if (sb.seg_.polyline_.empty()) {
    sb.append_node(node);
  }

  sb.append_node(node);

  utl::verify(sb.seg_.polyline_.size() == sb.seg_.osm_ids_.size(),
              "processed_segment: size missmatch");
  for (auto const& mask : sb.seg_.mask_) {
    utl::verify(sb.seg_.polyline_.size() == mask.size(),
                "processed_segment: mask size missmatch");
  }

  return std::move(sb.seg_);
}

void reconstruct_and_serialize_seqs(post_graph const& graph,
                                    db_builder& builder) {
  utl::verify(graph.originals_.size() == graph.segment_ids_.size(),
              "size mismatch");
  ml::scoped_timer timer("post_serializer: serialize_seqs");

  utl::parallel_for_run(graph.originals_.size(), [&](auto const i) {
    auto const seq = graph.originals_[i];
    auto const ids = graph.segment_ids_[i];

    auto const proc = utl::to_vec(ids, [&graph](auto const& id) {
      return reconstruct_segment(graph, id);
    });

    // check ...

    builder.add_seq(i, seq, proc);
  });
}

struct color_to_seq_seg_index {
  explicit color_to_seq_seg_index(post_graph const& graph) : graph_{graph} {
    for (auto i = 0UL; i < graph.segment_ids_.size(); ++i) {
      auto const& seg_ids = graph.segment_ids_.at(i);
      for (auto j = 0UL; j < seg_ids.size(); ++j) {
        color_to_seq_segs_.emplace_back(
            seg_ids.at(j).max_color_,
            seq_seg{static_cast<uint32_t>(i), static_cast<uint32_t>(j)});
      }
    }
    std::sort(
        begin(color_to_seq_segs_), end(color_to_seq_segs_),
        [](auto const& lhs, auto const& rhs) { return lhs.first < rhs.first; });
  }

  std::vector<seq_seg> decode_colors(std::vector<color_t> const& cs) const {
    return utl::to_vec(cs, [&](auto const& color) {
      auto const it = std::lower_bound(
          begin(color_to_seq_segs_), end(color_to_seq_segs_), color,
          [](auto const& lhs, auto const& rhs) { return lhs.first < rhs; });
      utl::verify(it != end(color_to_seq_segs_), "could not find seq_seg");
      return it->second;
    });
  }

  std::vector<uint32_t> get_classes(
      std::vector<seq_seg> const& seq_segs) const {
    std::vector<uint32_t> classes;
    for (auto const& seq_seg : seq_segs) {
      utl::concat(classes, graph_.originals_.at(seq_seg.sequence_).classes_);
    }
    return classes;
  }

  post_graph const& graph_;
  std::vector<std::pair<color_t, seq_seg>> color_to_seq_segs_;
};

void serialize_tiles(post_graph& graph, db_builder& builder) {
  ml::scoped_timer timer("post_serializer: serialize_tiles");
  color_to_seq_seg_index index{graph};

  std::vector<seq_seg> stub_seq_seqs;
  for (auto seq_idx = 0ULL; seq_idx < graph.originals_.size(); ++seq_idx) {
    auto const& seq = graph.originals_[seq_idx];
    for (auto const& info : seq.sequence_infos_) {
      if (info.source_spec_.router_ == source_spec::router::STUB &&
          info.between_stations_ &&
          (stub_seq_seqs.empty() ||
           stub_seq_seqs.back() != seq_seg{static_cast<uint32_t>(seq_idx),
                                           static_cast<uint32_t>(info.idx_)})) {
        stub_seq_seqs.emplace_back(seq_idx, info.idx_);
      }
    }
  }
  LOG(ml::info) << "found " << stub_seq_seqs.size()
                << " stub segments between stations.";

  for (auto const& ap : graph.atomic_paths_) {
    auto polyline =
        utl::to_vec(ap->path_, [](auto const& node) { return node->id_.pos_; });

    utl::verify(ap->path_.size() >= 2, "illformed atomic path (size = %zu)",
                ap->path_.size());
    auto edge = ap->path_.at(0)->find_edge_to(ap->path_.at(1));
    utl::verify(edge != nullptr, "missing edge in atomic path");

    auto seq_segs = index.decode_colors(edge->colors_);
    utl::erase_duplicates(seq_segs);
    auto classes = index.get_classes(seq_segs);
    utl::erase_duplicates(classes);

    auto const is_stub =
        std::any_of(begin(seq_segs), end(seq_segs), [&](auto const& ss) {
          auto const it =
              std::lower_bound(begin(stub_seq_seqs), end(stub_seq_seqs), ss);
          return it != end(stub_seq_seqs) && *it == ss;
        });

    builder.add_tile_feature(polyline, seq_segs, classes, is_stub);
  }

  for (auto const& pair : graph.atomic_pairs_) {
    auto polyline = geo::polyline{pair.first->id_.pos_, pair.second->id_.pos_};

    std::vector<seq_seg> seq_segs;
    auto const e1 = pair.first->find_edge_to(pair.second);
    if (e1 != nullptr) {
      utl::concat(seq_segs, index.decode_colors(e1->colors_));
    }
    auto const e2 = pair.second->find_edge_to(pair.first);
    if (e2 != nullptr) {
      utl::concat(seq_segs, index.decode_colors(e2->colors_));
    }
    utl::erase_duplicates(seq_segs);

    auto classes = index.get_classes(seq_segs);
    utl::erase_duplicates(classes);

    auto const is_stub =
        std::any_of(begin(seq_segs), end(seq_segs), [&](auto const& ss) {
          auto const it =
              std::lower_bound(begin(stub_seq_seqs), end(stub_seq_seqs), ss);
          return it != end(stub_seq_seqs) && *it == ss;
        });

    builder.add_tile_feature(polyline, seq_segs, classes, is_stub);
  }
}

void serialize_post_graph(post_graph& graph, db_builder& builder) {
  // check_out_colors(graph);

  reconstruct_and_serialize_seqs(graph, builder);
  serialize_tiles(graph, builder);
}

}  // namespace motis::path
