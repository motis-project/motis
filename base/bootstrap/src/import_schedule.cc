#include "motis/bootstrap/import_schedule.h"

#include "boost/filesystem.hpp"

#include "utl/pipes.h"

#include "motis/core/common/logging.h"
#include "motis/core/schedule/schedule_data_key.h"
#include "motis/module/clog_redirect.h"
#include "motis/module/context/motis_publish.h"
#include "motis/module/event_collector.h"
#include "motis/module/message.h"
#include "motis/bootstrap/import_settings.h"
#include "motis/loader/loader.h"

// namespace fs = boost::filesystem;
namespace mm = motis::module;

namespace motis::bootstrap {

void register_import_schedule(motis_instance& instance,
                              loader::loader_options const& dataset_opt,
                              std::string data_dir) {
  std::make_shared<mm::event_collector>(
      data_dir, "schedule", instance,
      [&, dataset_opt](std::map<std::string, mm::msg_ptr> const& dependencies) {
        auto const& msg = dependencies.at("SCHEDULE");

        auto const parsers = loader::parsers();
        using import::FileEvent;
        auto dataset_opt_cpy = dataset_opt;
        dataset_opt_cpy.dataset_.clear();
        dataset_opt_cpy.dataset_prefix_.clear();

        for (auto const* p : *motis_content(FileEvent, msg)->paths()) {
          if (p->tag()->str() == "schedule" &&
              std::any_of(begin(parsers), end(parsers),
                          [&](auto const& parser) {
                            return parser->applicable(p->path()->str());
                          })) {
            dataset_opt_cpy.dataset_.emplace_back(p->path()->str());
            dataset_opt_cpy.dataset_prefix_.emplace_back(p->options()->str());
          }
        }

        utl::verify(!dataset_opt_cpy.dataset_.empty(),
                    "import_schedule: dataset_opt.dataset_.empty()");

        cista::memory_holder memory;
        auto sched = loader::load_schedule(dataset_opt_cpy, memory);
        instance.shared_data_.emplace_data(
            SCHEDULE_DATA_KEY,
            schedule_data{std::move(memory), std::move(sched)});

        // (fs::path{p} / "schedule.raw").generic_string()),
        module::message_creator fbb;
        fbb.create_and_finish(MsgContent_ScheduleEvent,
                              import::CreateScheduleEvent(
                                  fbb,
                                  fbb.CreateString(""),  // TODO(sebastian)
                                  instance.sched().hash_)
                                  .Union(),
                              "/import", DestinationType_Topic);
        motis_publish(make_msg(fbb));
        return nullptr;
      })
      ->require("SCHEDULE", [](mm::msg_ptr const& msg) {
        if (msg->get()->content_type() != MsgContent_FileEvent) {
          return false;
        }

        auto any_applicable = false;
        using import::FileEvent;
        auto const parsers = loader::parsers();
        for (auto const* p : *motis_content(FileEvent, msg)->paths()) {
          if (p->tag()->str() != "schedule") {
            continue;
          }
          auto const& path = p->path()->str();
          auto const applicable = std::any_of(
              begin(parsers), end(parsers),
              [&](auto const& parser) { return parser->applicable(path); });

          if (!applicable) {
            std::clog << "import_schedule: no parser for " << path << "\n";
            for (auto const& parser : parsers) {
              std::clog << "missing files:\n";
              for (auto const& file : parser->missing_files(path)) {
                std::clog << "  " << file << "\n";
              }
            }
          }
          any_applicable = any_applicable || applicable;
        }
        return any_applicable;
      });
}

}  // namespace motis::bootstrap
