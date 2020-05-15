#pragma once

#include "geo/polyline.h"

#include "tiles/fixed/fixed_geometry.h"
#include "tiles/get_tile.h"

#include "motis/hash_map.h"

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
    uint32_t min_clasz_{std::numeric_limits<uint32_t>::max()};
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
  void add_extra(std::vector<geo::polyline>);

  void execute(path_database const&);
  void resolve_sequences_and_build_subqueries(lmdb::cursor&);
  void execute_subquery(tiles::tile_index_t, subquery&, lmdb::cursor&,
                        tiles::pack_handle const&);

  flatbuffers::Offset<PathSeqResponse> write_sequence(module::message_creator&,
                                                      path_database const&,
                                                      size_t index);

  flatbuffers::Offset<PathByTripIdBatchResponse> write_batch(
      module::message_creator&);

  int zoom_level_;

  std::vector<resolvable_sequence> sequences_;
  mcd::hash_map<tiles::tile_index_t, subquery> subqueries_;
  std::vector<std::unique_ptr<resolvable_feature>> extras_;
};

inline module::msg_ptr get_response(path_database const& db, size_t const index,
                                    int const zoom_level = -1) {
  path_database_query query{zoom_level};
  query.add_sequence(index);
  query.execute(db);

  module::message_creator mc;
  mc.create_and_finish(MsgContent_PathSeqResponse,
                       query.write_sequence(mc, db, index).Union());
  return make_msg(mc);
}

}  // namespace motis::path
