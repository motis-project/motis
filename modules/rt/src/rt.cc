#include "motis/rt/rt.h"

#include "utl/get_or_create.h"

#include "motis/core/schedule/serialization.h"

#include "motis/core/conv/trip_conv.h"

#include "motis/module/event_collector.h"
#include "motis/module/global_res_ids.h"

#include "motis/rt/error.h"
#include "motis/rt/rt_handler.h"

using namespace motis::ris;
using namespace motis::module;

namespace motis::rt {

rt::rt() : module("RT", "rt") {
  param(validate_graph_, "validate_graph",
        "validate routing graph after every rt update");
  param(validate_constant_graph_, "validate_constant_graph",
        "validate constant graph after every rt update");
  param(print_stats_, "print_stats", "print statistics after every rt update");
  param(enable_history_, "history", "enable message history for debugging");
}

rt::~rt() = default;

constexpr auto const DEFAULT_SCHEDULE_RES_ID =
    to_res_id(motis::module::global_res_id::SCHEDULE);

template <typename T>
ctx::res_id_t get_schedule_res_id(T const& m) {
  return m->schedule() == 0U ? DEFAULT_SCHEDULE_RES_ID
                             : static_cast<ctx::res_id_t>(m->schedule());
}

msg_ptr get_trip_history(schedule const& sched, rt_handler* rth,
                         RtMessageHistoryRequest const* req) {
  auto const trp = from_fbs(sched, req->trip());
  message_creator mc;
  std::vector<flatbuffers::Offset<motis::ris::RISMessage>> messages;
  if (rth != nullptr) {
    if (auto const it = rth->msg_history_.messages_.find(trp->trip_idx_);
        it != end(rth->msg_history_.messages_)) {
      auto const& buffers = it->second;
      messages.reserve(buffers.size());
      for (auto const& buf : buffers) {
        messages.emplace_back(motis_copy_table(RISMessage, mc, buf.get()));
      }
    }
  }
  mc.create_and_finish(
      MsgContent_RtMessageHistoryResponse,
      CreateRtMessageHistoryResponse(mc, mc.CreateVector(messages)).Union());
  return make_msg(mc);
}

void rt::init(motis::module::registry& reg) {
  reg.subscribe(
      "/ris/messages",
      [this](motis::module::msg_ptr const& msg) {
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
      [this](motis::module::msg_ptr const& msg) {
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
      [this](motis::module::msg_ptr const& msg) {
        auto schedule_res_id = DEFAULT_SCHEDULE_RES_ID;
        if (msg->get()->content_type() == MsgContent_RISSystemTimeChanged) {
          schedule_res_id =
              get_schedule_res_id(motis_content(RISSystemTimeChanged, msg));
        }
        // only used to lock the schedule, rt_handler already has a reference
        auto res_lock =
            lock_resources({{schedule_res_id, ctx::access_t::WRITE}});

        std::unique_lock lock{handler_mutex_};
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

  reg.register_op("/rt/dump",
                  [this](motis::module::msg_ptr const& msg) {
                    auto const m = motis_content(RtWriteGraphRequest, msg);
                    write_graph(m->path()->str(), get_sched());
                    return motis::module::msg_ptr{};
                  },
                  {::motis::module::kScheduleReadAccess});

  reg.register_op(
      "/rt/message_history",
      [this](msg_ptr const& msg) {
        auto const req = motis_content(RtMessageHistoryRequest, msg);
        auto const schedule_res_id = get_schedule_res_id(req);
        auto res_lock =
            lock_resources({{schedule_res_id, ctx::access_t::READ}});
        auto& sched = *res_lock.get<schedule_data>(schedule_res_id).schedule_;
        return get_trip_history(sched, get_rt_handler(schedule_res_id), req);
      },
      {});

  reg.register_op("/rt/metrics",
                  [this](msg_ptr const& msg) -> msg_ptr {
                    auto schedule_res_id = DEFAULT_SCHEDULE_RES_ID;
                    switch (msg->get()->content_type()) {
                      case MsgContent_RtMetricsRequest:
                        schedule_res_id = get_schedule_res_id(
                            motis_content(RtMetricsRequest, msg));
                        break;
                      case MsgContent_MotisNoMessage: break;
                      default:
                        throw std::system_error{
                            motis::module::error::unexpected_message_type};
                    }
                    auto const* handler = get_rt_handler(schedule_res_id);
                    if (handler == nullptr) {
                      throw std::system_error{error::schedule_not_found};
                    }
                    return get_metrics_api(handler->metrics_);
                  },
                  {});
}

void rt::import(motis::module::import_dispatcher& reg) {
  std::make_shared<motis::module::event_collector>(
      get_data_directory().generic_string(), "rt", reg,
      [this](motis::module::event_collector::dependencies_map_t const&,
             motis::module::event_collector::publish_fn_t const&) {
        import_successful_ = true;
      })
      ->require("SCHEDULE", [](motis::module::msg_ptr const& msg) {
        return msg->get()->content_type() == MsgContent_ScheduleEvent;
      });
}

bool rt::import_successful() const { return import_successful_; }

rt_handler& rt::get_or_create_rt_handler(schedule& sched,
                                         ctx::res_id_t const schedule_res_id) {
  std::lock_guard const guard{handler_mutex_};
  return *utl::get_or_create(handlers_, schedule_res_id,
                             [this, &sched, schedule_res_id]() {
                               return std::make_unique<rt_handler>(
                                   sched, schedule_res_id, validate_graph_,
                                   validate_constant_graph_, print_stats_,
                                   enable_history_);
                             })
              .get();
}

rt_handler* rt::get_rt_handler(ctx::res_id_t const schedule_res_id) {
  std::lock_guard const guard{handler_mutex_};
  if (auto const it = handlers_.find(schedule_res_id); it != end(handlers_)) {
    return it->second.get();
  } else {
    return nullptr;
  }
}

}  // namespace motis::rt
