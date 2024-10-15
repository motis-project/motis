#include "motis/rt/rt.h"

#include "utl/get_or_create.h"

#include "motis/core/schedule/serialization.h"
#include "motis/module/global_res_ids.h"

#include "motis/rt/rt_handler.h"

using namespace motis::ris;

namespace motis::rt {

rt::rt() : module("RT", "rt") {
  param(validate_graph_, "validate_graph",
        "validate routing graph after every rt update");
  param(validate_constant_graph_, "validate_constant_graph",
        "validate constant graph after every rt update");
  param(print_stats_, "print_stats", "print statistics after every rt update");
}

rt::~rt() = default;

constexpr auto const DEFAULT_SCHEDULE_RES_ID =
    to_res_id(motis::module::global_res_id::SCHEDULE);

void rt::init(motis::module::registry& reg) {
  auto const get_schedule_res_id = [](auto const& m) {
    return m->schedule() == 0U ? DEFAULT_SCHEDULE_RES_ID
                               : static_cast<ctx::res_id_t>(m->schedule());
  };

  reg.subscribe(
      "/ris/messages",
      [&](motis::module::msg_ptr const& msg) {
        auto const req = motis_content(RISBatch, msg);
        auto const schedule_res_id = get_schedule_res_id(req);
        auto res_lock =
            lock_resources({{schedule_res_id, ctx::access_t::WRITE}});
        auto& sched = *res_lock.get<schedule_data>(schedule_res_id).schedule_;
        return get_or_create_rt_handler(sched, schedule_res_id).update(msg);
      },
      {});

  reg.register_op(
      "/rt/single",
      [&](motis::module::msg_ptr const& msg) {
        return get_or_create_rt_handler(
                   const_cast<schedule&>(get_sched()),  // NOLINT
                   DEFAULT_SCHEDULE_RES_ID)
            .single(msg);
      },
      ctx::accesses_t{ctx::access_request{
          to_res_id(::motis::module::global_res_id::SCHEDULE),
          ctx::access_t::WRITE}});

  reg.subscribe(
      "/ris/system_time_changed",
      [&](motis::module::msg_ptr const& msg) {
        auto schedule_res_id = DEFAULT_SCHEDULE_RES_ID;
        if (msg->get()->content_type() == MsgContent_RISSystemTimeChanged) {
          schedule_res_id =
              get_schedule_res_id(motis_content(RISSystemTimeChanged, msg));
        }
        // only used to lock the schedule, rt_handler already has a reference
        auto res_lock =
            lock_resources({{schedule_res_id, ctx::access_t::WRITE}});

        std::unique_lock lock{handler_mutex};
        if (auto const it = handlers_.find(schedule_res_id);
            it != end(handlers_)) {
          lock.unlock();
          it->second->flush(msg);
          if (schedule_res_id != DEFAULT_SCHEDULE_RES_ID) {
            lock.lock();
            handlers_.erase(it);
          }
        }
        return nullptr;
      },
      {});

  reg.register_op("/rt/dump", [&](motis::module::msg_ptr const& msg) {
    auto const m = motis_content(RtWriteGraphRequest, msg);
    write_graph(m->path()->str(), get_sched());
    return motis::module::msg_ptr{};
  });
}

rt_handler& rt::get_or_create_rt_handler(schedule& sched,
                                         ctx::res_id_t const schedule_res_id) {
  std::lock_guard guard{handler_mutex};
  return *utl::get_or_create(handlers_, schedule_res_id, [&]() {
            return std::make_unique<rt_handler>(
                sched, schedule_res_id, validate_graph_,
                validate_constant_graph_, print_stats_);
          }).get();
}

}  // namespace motis::rt
