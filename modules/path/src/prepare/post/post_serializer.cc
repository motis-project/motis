#include "motis/path/prepare/post/post_serializer.h"

#include "utl/concat.h"
#include "utl/equal_ranges_linear.h"
#include "utl/erase_duplicates.h"
#include "utl/parallel_for.h"
#include "utl/to_vec.h"

#include "motis/core/common/logging.h"

#include "motis/path/prepare/db_builder.h"
#include "motis/path/prepare/osm_path.h"
#include "motis/path/prepare/post/color_to_seq_seg_index.h"
#include "motis/path/prepare/post/post_graph.h"

namespace ml = motis::logging;

namespace motis::path {

void serialize_geometry(post_graph& graph, db_builder& builder) {
  ml::scoped_timer timer("post_serializer: serialize_geometry");
  color_to_seq_seg_index index{graph};

  std::vector<seq_seg> stub_seq_seqs;
  for (auto seq_idx = 0ULL; seq_idx < graph.originals_->size(); ++seq_idx) {
    auto const& seq = graph.originals_->at(seq_idx);
    for (auto const& info : seq.sequence_infos_) {
      auto const ss = seq_seg{static_cast<uint32_t>(seq_idx),
                              static_cast<uint32_t>(info.idx_)};
      if (info.source_spec_.router_ == source_spec::router::STUB &&
          info.between_stations_ &&
          (stub_seq_seqs.empty() || stub_seq_seqs.back() != ss)) {
        stub_seq_seqs.push_back(ss);
      }
    }
  }
  LOG(ml::info) << "found " << stub_seq_seqs.size()
                << " stub segments between stations.";

  for (auto& ap : graph.atomic_paths_) {
    auto polyline =
        utl::to_vec(ap->path_, [](auto const& node) { return node->id_.pos_; });

    auto const& [seq_segs, classes] = index.resolve_atomic_path(*ap);

    auto const is_stub =
        std::any_of(begin(seq_segs), end(seq_segs), [&](auto const& ss) {
          auto const it =
              std::lower_bound(begin(stub_seq_seqs), end(stub_seq_seqs), ss);
          return it != end(stub_seq_seqs) && *it == ss;
        });

    utl::verify(!seq_segs.empty(), "post_serializer: abandoned feature");
    auto distance = -std::numeric_limits<float>::infinity();
    for (auto const& seq_seg : seq_segs) {
      distance =
          std::max(distance, graph.originals_->at(seq_seg.sequence_).distance_);
    }

    std::tie(ap->id_, ap->hint_) =
        builder.add_feature(polyline, seq_segs, classes, is_stub, distance);

    for (auto const& n : ap->path_) {
      ap->box_.extend(n->id_.pos_);
    }
  }
}

std::vector<std::pair<atomic_path*, bool>> reconstruct_path(
    post_segment_id const& segment_id, size_t const sanity_check_limit) {
  std::vector<std::pair<atomic_path*, bool>> paths;

  auto* node = segment_id.front_;
  auto color = segment_id.color_;

  if (node == nullptr) {
    return {};
  }

  for (auto i = 0ULL;; ++i) {
    if (i > sanity_check_limit) {
      LOG(logging::warn)
          << "reconstruct_path: sanity_check_limit reached, abort.";
      return {};
    }

    post_graph_edge* edge = nullptr;
    if (color < segment_id.max_color_) {
      edge = node->find_out_edge(color + 1);
    }

    if (edge != nullptr) {
      ++color;
    } else {
      // we are at back and have no color increment -> finish
      if (node == segment_id.back_) {
        break;
      }

      edge = node->find_out_edge(color);
    }

    if (edge == nullptr) {
      break;
    }

    utl::verify(edge->atomic_path_ != nullptr,
                "have an edge without atomic_path");
    paths.emplace_back(edge->atomic_path_, edge->atomic_path_forward_);

    node = edge->atomic_path_forward_ ? edge->atomic_path_->to_
                                      : edge->atomic_path_->from_;
  }
  utl::verify(!paths.empty(), "reconstruct_path: failure");

  return paths;
}

void reconstruct_and_serialize_seqs(post_graph const& graph,
                                    db_builder& builder) {
  utl::verify(graph.originals_->size() == graph.segment_ids_.size(),
              "size mismatch");
  ml::scoped_timer timer("post_serializer: serialize_seqs");

  utl::parallel_for_run(graph.originals_->size(), [&](auto const i) {
    auto const& seq = graph.originals_->at(i);
    auto const& ids = graph.segment_ids_.at(i);

    std::vector<geo::box> boxes;
    std::vector<std::vector<int64_t>> feature_ids;
    std::vector<std::vector<uint64_t>> hints_rle;
    for (auto j = 0ULL; j < seq.paths_.size(); ++j) {
      auto const atomic_paths =
          reconstruct_path(ids.at(j), seq.paths_.at(j).size());

      boxes.emplace_back();
      for (auto const& ap : atomic_paths) {
        boxes.back().extend(ap.first->box_);
      }

      feature_ids.emplace_back(
          utl::to_vec(atomic_paths, [](auto const& pair) -> int64_t {
            return pair.first->id_ * (pair.second ? 1 : -1);
          }));

      hints_rle.emplace_back();
      utl::equal_ranges_linear(
          atomic_paths,
          [](auto const& lhs, auto const& rhs) {
            return lhs.first->hint_ == rhs.first->hint_;
          },
          [&](auto lb, auto ub) {
            hints_rle.back().emplace_back(lb->first->hint_);
            hints_rle.back().emplace_back(std::distance(lb, ub));
          });
    }

    builder.add_seq(i, seq, boxes, feature_ids, hints_rle);
  });
}

void serialize_post_graph(post_graph& graph, db_builder& builder) {
  serialize_geometry(graph, builder);
  reconstruct_and_serialize_seqs(graph, builder);
}

}  // namespace motis::path
