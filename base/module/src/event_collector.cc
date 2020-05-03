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
          fbb.CreateVector(utl::to_vec(
              waiting_for_, [&](auto&& w) { return fbb.CreateString(w); })),
          status, progress)
          .Union(),
      "/import", DestinationType_Topic);
  ctx::await_all(motis_publish(make_msg(fbb)));
}

void event_collector::require(std::string name,
                              std::function<bool(msg_ptr)> matcher) {
  auto const matcher_it =
      matchers_.emplace(dependency_matcher{name, std::move(matcher)});
  waiting_for_.emplace(name);
  reg_.subscribe("/import",
                 [&, matcher_it = matcher_it.first, name,
                  self = shared_from_this()](msg_ptr const& msg) -> msg_ptr {
                   // Dummy message asking for initial status.
                   // Send "waiting for" dependencies list.
                   if (msg->get()->content_type() == MsgContent_MotisSuccess) {
                     update_status(motis::import::Status_WAITING);
                     return nullptr;
                   }

                   if (!(matcher_it->matcher_fn_)(msg)) {
                     return nullptr;  // Not the right message type.
                   }

                   dependencies_[name] = msg;
                   waiting_for_.erase(name);
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