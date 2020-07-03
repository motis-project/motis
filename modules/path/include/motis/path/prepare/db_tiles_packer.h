#pragma once

#include "tiles/db/feature_pack_quadtree.h"

namespace motis::path {

struct db_tiles_packer : public tiles::quadtree_feature_packer {
  struct serialized_feature {
    uint32_t id_, offset_;
    service_class min_clasz_;
  };

  db_tiles_packer(geo::tile root,
                  tiles::shared_metadata_coder const& metadata_coder,
                  size_t layer_id)
      : tiles::quadtree_feature_packer{root, metadata_coder},
        layer_id_{layer_id} {
    packer_.register_segment(kPathIndexId);
  }

  uint32_t serialize_and_append_span(
      tiles::quadtree_feature_packer::quadtree_feature_it begin,
      tiles::quadtree_feature_packer::quadtree_feature_it end) override {
    uint32_t offset = packer_.buf_.size();
    for (auto it = begin; it != end; ++it) {
      auto const& f = it->feature_;

      if (f.layer_ == layer_id_) {
        auto const it =
            std::find_if(std::begin(f.meta_), std::end(f.meta_),
                         [](auto const& m) { return m.key_ == "min_class"; });
        utl::verify(it != std::end(f.meta_),
                    "db_tiles_packer: missing min_clasz");

        serialized_features_.push_back(
            {static_cast<uint32_t>(f.id_),
             static_cast<uint32_t>(packer_.buf_.size()),
             static_cast<service_class>(
                 tiles::read<int64_t>(it->value_.data(), 1))});
      }

      packer_.append_feature(serialize_feature(f, metadata_coder_, false));
    }
    packer_.append_span_end();
    return offset;
  }

  void make_index() {
    std::sort(begin(serialized_features_), end(serialized_features_),
              [](auto const& lhs, auto const& rhs) {
                return std::tie(lhs.min_clasz_, lhs.id_) <
                       std::tie(rhs.min_clasz_, rhs.id_);
              });

    std::vector<uint32_t> feature_counts(
        static_cast<service_class_t>(service_class::NUM_CLASSES), 0U);
    utl::equal_ranges_linear(
        serialized_features_,
        [](auto const& lhs, auto const& rhs) {
          return lhs.min_clasz_ == rhs.min_clasz_;
        },
        [&](auto lb, auto ub) {
          utl::verify(lb->min_clasz_ < service_class::NUM_CLASSES,
                      "db_tiles_packer: min_clasz to big");
          feature_counts[static_cast<service_class_t>(lb->min_clasz_)] =
              std::distance(lb, ub);
        });

    packer_.update_segment_offset(kPathIndexId, packer_.buf_.size());
    for (auto const count : feature_counts) {
      tiles::append<uint32_t>(packer_.buf_, count);
    }

    for (auto const& sf : serialized_features_) {
      tiles::append<uint32_t>(packer_.buf_, sf.id_);
      tiles::append<uint32_t>(packer_.buf_, sf.offset_);
    }
  }

  size_t layer_id_;
  std::vector<serialized_feature> serialized_features_;
};

}  // namespace motis::path
