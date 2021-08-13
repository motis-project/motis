#include "motis/revise/revise.h"

#include "boost/program_options.hpp"

#include "motis/core/journey/journeys_to_message.h"
#include "motis/core/journey/message_to_journeys.h"
#include "motis/revise/update_journey.h"

using namespace motis::module;

namespace motis::revise {

revise::revise() : module("Revise", "revise") {}

revise::~revise() = default;

void revise::init(motis::module::registry& reg) {
  reg.register_op("/revise", [&](msg_ptr const& m) { return update(m); });
}

msg_ptr revise::update(msg_ptr const& msg) {
  switch (msg->get()->content_type()) {
    case MsgContent_Connection: return update(motis_content(Connection, msg));
    case MsgContent_ReviseRequest:
      return update(motis_content(ReviseRequest, msg));
    default: throw std::system_error(error::unexpected_message_type);
  }
}

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
