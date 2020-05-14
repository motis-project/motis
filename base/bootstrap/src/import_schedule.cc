#include "motis/bootstrap/import_schedule.h"

#include "boost/filesystem.hpp"

#include "motis/core/common/logging.h"
#include "motis/module/clog_redirect.h"
#include "motis/module/context/motis_publish.h"
#include "motis/loader/loader.h"

namespace fs = boost::filesystem;

namespace motis::bootstrap {

module::msg_ptr import_schedule(module_settings const& module_opt,
                                loader::loader_options const& dataset_opt,
                                module::msg_ptr const& msg,
                                motis_instance& instance) {
  if (msg->get()->content_type() != MsgContent_FileEvent) {
    return nullptr;
  }

  using import::FileEvent;
  auto const path = fs::path{motis_content(FileEvent, msg)->path()->str()};
  if (!fs::is_directory(path)) {
    return nullptr;
  }

  auto dataset_opt_cpy = dataset_opt;
  dataset_opt_cpy.dataset_ = path.generic_string();

  cista::memory_holder memory;
  auto sched = loader::load_schedule(dataset_opt_cpy, memory);
  instance.shared_data_.emplace_data(
      "schedule", schedule_data{std::move(memory), std::move(sched)});

  module::message_creator fbb;
  fbb.create_and_finish(
      MsgContent_ScheduleEvent,
      import::CreateScheduleEvent(
          fbb,
          fbb.CreateString((fs::path{path} / "schedule.raw").generic_string()),
          instance.sched_->hash_)
          .Union(),
      "/import", DestinationType_Topic);
  ctx::await_all(motis_publish(make_msg(fbb)));

  return nullptr;
}

}  // namespace motis::bootstrap
