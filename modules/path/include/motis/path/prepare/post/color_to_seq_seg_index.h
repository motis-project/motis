#pragma once

#include "utl/enumerate.h"
#include "utl/erase_duplicates.h"
#include "utl/to_vec.h"

#include "motis/path/prepare/db_builder.h"
#include "motis/path/prepare/post/post_graph.h"

namespace motis::path {

struct color_to_seq_seg_index {
  explicit color_to_seq_seg_index(post_graph const& graph) : graph_{graph} {
    for (auto const& [i, sequence] : utl::enumerate(graph.segment_ids_)) {
      for (auto const& [j, segment] : utl::enumerate(sequence)) {
        if (!segment.valid()) {
          continue;
        }

        for (color_t c = segment.color_; c <= segment.max_color_; ++c) {
          color_to_seq_segs_.emplace_back(
              c, seq_seg{static_cast<uint32_t>(i), static_cast<uint32_t>(j)});
        }
      }
    }

    std::sort(
        begin(color_to_seq_segs_), end(color_to_seq_segs_),
        [](auto const& lhs, auto const& rhs) { return lhs.first < rhs.first; });
  }

  std::pair<std::vector<seq_seg>, mcd::vector<service_class>>
  resolve_atomic_path(atomic_path const& ap) const {
    auto const& p = ap.path_;
    utl::verify(p.size() >= 2, "illformed atomic path (size = {})", p.size());

    auto fwd_edge = p.at(0)->find_edge_to(p.at(1));
    utl::verify(fwd_edge != nullptr, "missing edge in atomic path");

    auto seq_segs = decode_colors(fwd_edge->colors_);

    auto bwd_edge = p.at(p.size() - 1)->find_edge_to(p.at(p.size() - 2));
    if (bwd_edge != nullptr) {
      utl::concat(seq_segs, decode_colors(bwd_edge->colors_));
    }
    utl::erase_duplicates(seq_segs);

    auto classes = get_classes(seq_segs);
    utl::erase_duplicates(classes);

    return std::make_pair(std::move(seq_segs), std::move(classes));
  }

  std::vector<seq_seg> decode_colors(std::vector<color_t> const& cs) const {
    return utl::to_vec(cs, [&](auto const& color) {
      auto const it = std::lower_bound(
          begin(color_to_seq_segs_), end(color_to_seq_segs_), color,
          [](auto const& lhs, auto const& rhs) { return lhs.first < rhs; });
      utl::verify(it != end(color_to_seq_segs_) && it->first == color,
                  "could not find seq_seg");
      return it->second;
    });
  }

  mcd::vector<service_class> get_classes(
      std::vector<seq_seg> const& seq_segs) const {
    mcd::vector<service_class> classes;
    for (auto const& seq_seg : seq_segs) {
      utl::concat(classes, graph_.originals_->at(seq_seg.sequence_).classes_);
    }
    return classes;
  }

  post_graph const& graph_;
  std::vector<std::pair<color_t, seq_seg>> color_to_seq_segs_;
};

}  // namespace motis::path
