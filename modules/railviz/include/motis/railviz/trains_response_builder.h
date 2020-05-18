#pragma once

#include "motis/core/schedule/schedule.h"

#include "motis/module/message.h"

namespace motis::railviz {

struct trains_response_builder {
  trains_response_builder(schedule const& sched, int zoom_level)
      : sched_{sched}, zoom_level_{zoom_level} {}

  void add_train_full(ev_key k);
  void add_ev_key(ev_key k);

  [[nodiscard]] module::msg_ptr resolve_paths();

  [[nodiscard]] flatbuffers::Offset<Train> write_railviz_train(
      module::message_creator&, trip const*, size_t const section_index,
      flatbuffers::Vector<int64_t> const* polyline_indices);

  [[nodiscard]] module::msg_ptr finish();

  schedule const& sched_;
  int zoom_level_;
  std::vector<std::pair<trip const*, size_t>> queries_;
  std::vector<uint32_t> station_indices_;
};

}  // namespace motis::railviz
