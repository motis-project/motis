#pragma once

#include "geo/polyline.h"

#include "tiles/fixed/fixed_geometry.h"
#include "tiles/get_tile.h"

#include "motis/hash_map.h"

#include "motis/core/schedule/connection.h"
#include "motis/module/message.h"

#include "motis/path/path_database.h"

#include "motis/path/fbs/InternalDbSequence_generated.h"

namespace motis::path {

struct path_database_query {
  static constexpr auto kInvalidResponseId =
      std::numeric_limits<uint64_t>::max();
  static constexpr auto kExtraSequenceIndex =
      std::numeric_limits<size_t>::max();

  struct resolvable_feature {
    uint32_t use_count() const { return fwd_use_count_ + bwd_use_count_; }

    uint64_t feature_id_{0};
    uint64_t response_id_{kInvalidResponseId};
    uint32_t fwd_use_count_{0};
    uint32_t bwd_use_count_{0};
    service_class min_clasz_{service_class::OTHER};  // "max"
    bool is_resolved_{false};
    bool is_reversed_{false};
    bool is_extra_{false};
    tiles::fixed_geometry geometry_;
  };

  struct resolvable_sequence {
    resolvable_sequence(size_t index, std::vector<size_t> segment_indices)
        : index_{index}, segment_indices_{std::move(segment_indices)} {}

    size_t index_;
    std::vector<size_t> segment_indices_;
    std::vector<std::vector<std::pair<bool, resolvable_feature*>>>
        segment_features_;
  };

  struct subquery {
    mcd::hash_map<uint64_t, resolvable_feature*> map_;
    std::vector<std::unique_ptr<resolvable_feature>> mem_;
  };

  explicit path_database_query(int const zoom_level = -1)
      : zoom_level_{zoom_level} {}

  void add_sequence(size_t index, std::vector<size_t> segment_indices = {});
  void add_extra(std::vector<geo::polyline> const&);

  void execute(path_database const&);
  void resolve_sequences_and_build_subqueries(lmdb::cursor&);
  void execute_subquery(tiles::tile_key_t, subquery&, lmdb::cursor&,
                        tiles::pack_handle const&);

  flatbuffers::Offset<PathSeqResponse> write_sequence(module::message_creator&,
                                                      path_database const&,
                                                      size_t index);

  flatbuffers::Offset<PathByTripIdBatchResponse> write_batch(
      module::message_creator&);

  void write_batch(
      module::message_creator& mc,
      std::vector<std::vector<int64_t>>& fbs_segments,
      std::vector<flatbuffers::Offset<flatbuffers::String>>& fbs_polylines,
      std::vector<uint64_t>& fbs_extras);

  int zoom_level_;

  std::vector<resolvable_sequence> sequences_;
  mcd::hash_map<tiles::tile_key_t, subquery> subqueries_;
  std::vector<std::unique_ptr<resolvable_feature>> extras_;
};

module::msg_ptr get_response(path_database const&, size_t index,
                             int zoom_level = -1);

}  // namespace motis::path
