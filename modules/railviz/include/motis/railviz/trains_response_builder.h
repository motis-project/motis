#pragma once

#include "motis/core/schedule/schedule.h"

#include "motis/module/message.h"

#include "motis/path/path_data.h"

#include "motis/railviz/train.h"

namespace motis::railviz {

struct trains_response_builder {
  struct query {
    MAKE_COMPARABLE();

    trip const* trp_;
    int section_index_;
    train train_;
  };

  trains_response_builder(schedule const& sched,
                          path::path_data const* path_data, int zoom_level)
      : sched_{sched}, path_data_{path_data}, zoom_level_{zoom_level} {}

  void add_train_full(ev_key);
  void add_train(train);

  void resolve_paths();
  void resolve_paths_fallback();
  void resolve_paths_fallback(size_t query_idx);

  flatbuffers::Offset<Train> write_railviz_train(
      query const&, std::vector<int64_t> const& polyline_indices);

  module::msg_ptr finish();

  schedule const& sched_;
  path::path_data const* path_data_;

  int zoom_level_;
  std::vector<query> queries_;
  std::vector<uint32_t> station_indices_;

  module::message_creator mc_;
  std::vector<std::vector<int64_t>> fbs_segments_;
  std::vector<flatbuffers::Offset<flatbuffers::String>> fbs_polylines_;
  std::vector<uint64_t> fbs_extras_;

  mcd::hash_map<std::pair<uint32_t, uint32_t>, size_t> fallback_indices_;
};

}  // namespace motis::railviz
