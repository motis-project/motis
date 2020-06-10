#include "motis/bootstrap/import_schedule.h"

#include "boost/filesystem.hpp"

#include "motis/core/common/logging.h"
#include "motis/core/schedule/schedule_data_key.h"
#include "motis/module/clog_redirect.h"
#include "motis/module/context/motis_publish.h"
#include "motis/loader/loader.h"

namespace fs = boost::filesystem;

namespace motis::bootstrap {

module::msg_ptr import_schedule(loader::loader_options const& dataset_opt,
                                module::msg_ptr const& msg,
                                motis_instance& instance) {
  if (msg->get()->content_type() != MsgContent_FileEvent) {
    return nullptr;
  }

  using import::FileEvent;
  for (auto const* p : *motis_content(FileEvent, msg)->paths()) {
    auto const& path = fs::path{p->str()};
    if (!fs::is_directory(path)) {
      continue;
    }

    auto dataset_opt_cpy = dataset_opt;
    dataset_opt_cpy.dataset_ = path.generic_string();

    try {
      cista::memory_holder memory;
      auto sched = loader::load_schedule(dataset_opt_cpy, memory);
      instance.shared_data_.emplace_data(
          SCHEDULE_DATA_KEY,
          schedule_data{std::move(memory), std::move(sched)});

      module::message_creator fbb;
      fbb.create_and_finish(
          MsgContent_ScheduleEvent,
          import::CreateScheduleEvent(
              fbb,
              fbb.CreateString(
                  (fs::path{path} / "schedule.raw").generic_string()),
              instance.sched().hash_)
              .Union(),
          "/import", DestinationType_Topic);
      motis_publish(make_msg(fbb));

    } catch (std::exception const&) {
      continue;
    }
    break;
  }
  return nullptr;
}

}  // namespace motis::bootstrap
