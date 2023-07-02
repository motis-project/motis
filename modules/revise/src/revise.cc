#include "motis/revise/revise.h"

#include "boost/program_options.hpp"

#include "motis/core/journey/journeys_to_message.h"
#include "motis/core/journey/message_to_journeys.h"
#include "motis/module/event_collector.h"
#include "motis/revise/update_journey.h"

using namespace motis::module;

namespace motis::revise {

revise::revise() : module("Revise", "revise") {}

revise::~revise() = default;

void revise::init(motis::module::registry& reg) {
  reg.register_op("/revise", [&](msg_ptr const& m) { return update(m); },
                  {kScheduleReadAccess});
}

msg_ptr revise::update(msg_ptr const& msg) {
  switch (msg->get()->content_type()) {
    case MsgContent_Connection: return update(motis_content(Connection, msg));
    case MsgContent_ReviseRequest:
      return update(motis_content(ReviseRequest, msg));
    default: throw std::system_error(error::unexpected_message_type);
  }
}

void revise::import(motis::module::import_dispatcher& reg) {
  std::make_shared<event_collector>(
      get_data_directory().generic_string(), "revise", reg,
      [this](event_collector::dependencies_map_t const&,
             event_collector::publish_fn_t const&) {
        import_successful_ = true;
      })
      ->require("SCHEDULE", [](msg_ptr const& msg) {
        return msg->get()->content_type() == MsgContent_ScheduleEvent;
      });
}

bool revise::import_successful() const { return import_successful_; }

msg_ptr revise::update(Connection const* con) {
  message_creator fbb;
  fbb.create_and_finish(
      MsgContent_Connection,
      to_connection(fbb, update_journey(get_sched(), convert(con))).Union());
  return make_msg(fbb);
}

msg_ptr revise::update(ReviseRequest const* req) {
  message_creator fbb;
  fbb.create_and_finish(
      MsgContent_ReviseResponse,
      CreateReviseResponse(fbb, fbb.CreateVector(utl::to_vec(
                                    *req->connections(),
                                    [&](Connection const* con) {
                                      return to_connection(
                                          fbb, update_journey(get_sched(),
                                                              convert(con)));
                                    })))
          .Union());
  return make_msg(fbb);
}

}  // namespace motis::revise
