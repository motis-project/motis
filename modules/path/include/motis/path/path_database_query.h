#pragma once

#include "tiles/fixed/fixed_geometry.h"
#include "tiles/get_tile.h"

#include "motis/hash_map.h"

#include "motis/module/message.h"

#include "motis/path/path_database.h"

#include "motis/path/fbs/InternalDbSequence_generated.h"

namespace motis::path {

struct path_database_query {
  struct resolvable_feature {
    uint64_t feature_id_{0};
    uint64_t response_id_{std::numeric_limits<uint64_t>::max()};
    uint64_t include_count_{0};
    uint32_t min_clasz_{std::numeric_limits<uint32_t>::max()};
    bool is_resolved_{false};
    tiles::fixed_geometry geometry_;
  };

  struct resolvable_sequence {
    size_t index_;
    std::vector<std::vector<std::pair<bool, resolvable_feature*>>>
        segment_features_;
  };

  struct subquery {
    mcd::hash_map<uint64_t, resolvable_feature*> map_;
    std::vector<std::unique_ptr<resolvable_feature>> mem_;
  };

  explicit path_database_query(int const zoom_level = -1)
      : zoom_level_{zoom_level} {}

  void add_sequence(size_t index);

  void execute(path_database const&, tiles::render_ctx const&);
  void resolve_sequences_and_build_subqueries(lmdb::cursor&);
  void execute_subquery(tiles::tile_index_t, subquery&, size_t layer_idx,  //
                        lmdb::cursor&, tiles::pack_handle const&,
                        tiles::render_ctx const&);

  flatbuffers::Offset<PathSeqResponse> write_sequence(module::message_creator&,
                                                      path_database const&,
                                                      size_t index);

  flatbuffers::Offset<PathSeqResponse> write_batch(module::message_creator&);

  int zoom_level_;

  std::vector<resolvable_sequence> sequences_;
  mcd::hash_map<tiles::tile_index_t, subquery> subqueries_;
};

inline module::msg_ptr get_response(path_database const& db,
                                    tiles::render_ctx const& render_ctx,
                                    size_t const index,
                                    int const zoom_level = -1) {
  path_database_query query{zoom_level};
  query.add_sequence(index);
  query.execute(db, render_ctx);

  module::message_creator mc;
  mc.create_and_finish(MsgContent_PathSeqResponse,
                       query.write_sequence(mc, db, index).Union());
  return make_msg(mc);
}

}  // namespace motis::path
