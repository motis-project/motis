#include "motis/bootstrap/import_schedule.h"

#include "boost/filesystem.hpp"

#include "motis/core/common/logging.h"
#include "motis/module/clog_redirect.h"
#include "motis/module/context/motis_publish.h"
#include "motis/loader/loader.h"

namespace fs = boost::filesystem;
using motis::module::log_streambuf;

namespace motis::bootstrap {

motis::module::msg_ptr import_schedule(
    import_settings const& import_opt, module_settings const& module_opt,
    loader::loader_options const& dataset_opt,
    motis::module::msg_ptr const& msg, motis_instance& instance) {
  if (msg->get()->content_type() != MsgContent_FileEvent) {
    return nullptr;
  }

  using motis::import::FileEvent;
  auto const path = fs::path{motis_content(FileEvent, msg)->path()->str()};
  if (!fs::is_directory(path)) {
    return nullptr;
  }

  try {
    log_streambuf redirect{
        "schedule", (fs::path{import_opt.data_directory_} / "schedule_log.txt")
                        .generic_string()
                        .c_str()};

    auto dataset_opt_cpy = dataset_opt;
    dataset_opt_cpy.dataset_ = path.generic_string();
    instance.schedule_ =
        loader::load_schedule(dataset_opt_cpy, instance.schedule_buf_);
    instance.sched_ = instance.schedule_.get();

    for (auto const& module : instance.modules_) {
      if (!module_opt.is_module_active(module->prefix())) {
        continue;
      }
      module->set_context(*instance.schedule_);
    }

    motis::module::message_creator fbb;
    fbb.create_and_finish(
        MsgContent_ScheduleEvent,
        motis::import::CreateScheduleEvent(
            fbb,
            fbb.CreateString(
                (fs::path{path} / "schedule.raw").generic_string()),
            instance.sched_->hash_)
            .Union(),
        "/import", DestinationType_Topic);
    ctx::await_all(motis_publish(make_msg(fbb)));
  } catch (std::exception const& e) {
    LOG(logging::error) << e.what();
  }

  return nullptr;
}

}  // namespace motis::bootstrap
