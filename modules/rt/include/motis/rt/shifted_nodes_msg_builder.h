#pragma once

#include <vector>

#include "motis/core/schedule/delay_info.h"
#include "motis/core/schedule/schedule.h"
#include "motis/module/message.h"

namespace motis::rt {

struct shifted_nodes_msg_builder {
  explicit shifted_nodes_msg_builder(motis::module::message_creator& fbb,
                                     schedule const&);

  void add(delay_info const* di);
  void finish(std::vector<flatbuffers::Offset<RtUpdate>>&);
  bool empty() const;
  std::size_t size() const;

private:
  flatbuffers::Offset<RtUpdate> build_shifted_node(delay_info const* di);

  motis::module::message_creator& fbb_;
  schedule const& sched_;
  std::set<delay_info const*> delays_;
};

}  // namespace motis::rt
