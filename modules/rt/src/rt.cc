#include "motis/rt/rt.h"

#include "boost/program_options.hpp"

#include "motis/core/schedule/serialization.h"

#include "motis/rt/rt_handler.h"

namespace motis::rt {

rt::rt() : module("RT", "rt") {
  param(validate_graph_, "validate_graph",
        "validate routing graph after every rt update");
  param(validate_constant_graph_, "validate_constant_graph",
        "validate constant graph after every rt update");
  param(print_stats_, "print_stats", "print statistics after every rt update");
}

rt::~rt() = default;

void rt::init(motis::module::registry& reg) {
  handler_ = std::make_unique<rt_handler>(
      *const_cast<schedule*>(&get_sched()),  // NOLINT
      validate_graph_, validate_constant_graph_, print_stats_);

  reg.subscribe(
      "/ris/messages",
      [&](motis::module::msg_ptr const& msg) { return handler_->update(msg); },
      ctx::access_t::WRITE);
  reg.register_op(
      "/rt/single",
      [&](motis::module::msg_ptr const& msg) { return handler_->single(msg); },
      ctx::access_t::WRITE);
  reg.subscribe(
      "/ris/system_time_changed",
      [&](motis::module::msg_ptr const& msg) {
        handler_->flush(msg);
        return nullptr;
      },
      ctx::access_t::WRITE);
  reg.register_op("/rt/dump", [&](motis::module::msg_ptr const& msg) {
    auto const m = motis_content(RtWriteGraphRequest, msg);
    write_graph(m->path()->str(), get_sched());
    return motis::module::msg_ptr{};
  });
}

}  // namespace motis::rt
