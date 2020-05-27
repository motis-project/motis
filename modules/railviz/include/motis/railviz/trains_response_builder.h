#pragma once

#include "motis/core/schedule/schedule.h"

#include "motis/module/message.h"

#include "motis/railviz/train.h"

namespace motis::railviz {

struct trains_response_builder {
  struct query {
    MAKE_COMPARABLE();

    trip const* trp_;
    int section_index_;
    train train_;
  };

  trains_response_builder(schedule const& sched, int zoom_level)
      : sched_{sched}, zoom_level_{zoom_level} {}

  void add_train_full(ev_key);
  void add_train(train);

  [[nodiscard]] module::msg_ptr resolve_paths();

  [[nodiscard]] flatbuffers::Offset<Train> write_railviz_train(
      module::message_creator&, query const&,
      flatbuffers::Vector<int64_t> const* polyline_indices);

  [[nodiscard]] module::msg_ptr finish();

  schedule const& sched_;
  int zoom_level_;
  std::vector<query> queries_;
  std::vector<uint32_t> station_indices_;
};

}  // namespace motis::railviz
