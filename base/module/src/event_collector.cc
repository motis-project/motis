#include "motis/module/event_collector.h"

#include "motis/module/context/motis_publish.h"

namespace motis::module {

event_collector::event_collector(std::string name, registry& reg,
                                 import_op_t op)
    : module_name_{std::move(name)}, reg_{reg}, op_{std::move(op)} {}

void event_collector::update_status(motis::import::Status const status,
                                    uint8_t const progress) {
  message_creator fbb;
  fbb.create_and_finish(
      MsgContent_StatusUpdate,
      motis::import::CreateStatusUpdate(
          fbb, fbb.CreateString(module_name_),
          fbb.CreateVector(utl::to_vec(waiting_for_,
                                       [&](MsgContent msg_type) {
                                         return fbb.CreateString(
                                             EnumNameMsgContent(msg_type));
                                       })),
          status, progress)
          .Union(),
      "/import", DestinationType_Topic);
  ctx::await_all(motis_publish(make_msg(fbb)));
}

void event_collector::listen(MsgContent const msg_type) {
  waiting_for_.emplace(msg_type);
  reg_.subscribe(
      "/import",
      [&, msg_type, self = shared_from_this()](msg_ptr const& msg) -> msg_ptr {
        if (msg->get()->content_type() != msg_type) {
          return nullptr;  // Not the right message type.
        }

        dependencies_[msg_type] = msg;
        waiting_for_.erase(msg_type);
        if (!waiting_for_.empty()) {
          return nullptr;  // Still waiting for a message.
        }

        // All messages arrived -> start.
        update_status(motis::import::Status_RUNNING);
        op_(dependencies_);
        update_status(motis::import::Status_FINISHED);

        return nullptr;
      });
}

}  // namespace motis::module