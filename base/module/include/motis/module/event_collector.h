#pragma once

#include <cinttypes>
#include <functional>
#include <map>
#include <string>

#include "motis/module/message.h"
#include "motis/module/registry.h"

namespace motis::module {

struct event_collector : std::enable_shared_from_this<event_collector> {
  using dependencies_map_t = std::map<MsgContent, msg_ptr>;
  using import_op_t = std::function<void(dependencies_map_t const&)>;

  event_collector(std::string name, registry& reg, import_op_t op);

  void listen(MsgContent msg);

private:
  void update_status(motis::import::Status, uint8_t progress = 0U);

  std::string module_name_;
  registry& reg_;
  import_op_t op_;
  dependencies_map_t dependencies_;
  std::set<MsgContent> waiting_for_;
};

}  // namespace motis::module