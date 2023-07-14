#include "motis/bootstrap/import_schedule.h"

#include "utl/pipes.h"

#include "motis/core/common/logging.h"
#include "motis/core/schedule/station_lookup.h"
#include "motis/module/clog_redirect.h"
#include "motis/module/context/motis_publish.h"
#include "motis/module/event_collector.h"
#include "motis/module/message.h"
#include "motis/bootstrap/import_settings.h"
#include "motis/loader/loader.h"

namespace mm = motis::module;

namespace motis::bootstrap {

void register_import_schedule(motis_instance& instance,
                              mm::import_dispatcher& reg,
                              loader::loader_options const& dataset_opt,
                              std::string const& data_dir) {
  if (dataset_opt.no_schedule_) {
    return;
  }

  std::make_shared<mm::event_collector>(
      data_dir, "schedule", reg,
      [&, dataset_opt](
          mm::event_collector::dependencies_map_t const& dependencies,
          mm::event_collector::publish_fn_t const& publish) {
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
        auto sched = loader::load_schedule(dataset_opt_cpy, memory, data_dir);
        std::shared_ptr<station_lookup> station_lookup =
            std::make_shared<schedule_station_lookup>(*sched);
        instance.emplace_data(motis::module::to_res_id(
                                  motis::module::global_res_id::STATION_LOOKUP),
                              std::move(station_lookup));
        instance.emplace_data(
            motis::module::to_res_id(motis::module::global_res_id::SCHEDULE),
            schedule_data{std::move(memory), std::move(sched)});
        {
          mm::message_creator fbb;
          fbb.create_and_finish(
              MsgContent_ScheduleEvent,
              import::CreateScheduleEvent(
                  fbb,
                  fbb.CreateVector(utl::to_vec(
                      dataset_opt_cpy.dataset_,
                      [&, i = 0](auto const&) mutable {
                        return fbb.CreateString(
                            dataset_opt_cpy.fbs_schedule_path(data_dir, i++));
                      })),
                  fbb.CreateVector(utl::to_vec(dataset_opt_cpy.dataset_prefix_,
                                               [&](auto const& prefix) {
                                                 return fbb.CreateString(
                                                     prefix);
                                               })),
                  instance.sched().hash_)
                  .Union(),
              "/import", DestinationType_Topic);
          publish(make_msg(fbb));
        }
        {
          mm::message_creator fbb;
          fbb.create_and_finish(MsgContent_StationsEvent,
                                motis::import::CreateStationsEvent(fbb).Union(),
                                "/import", DestinationType_Topic);
          publish(make_msg(fbb));
        }
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
