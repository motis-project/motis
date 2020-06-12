#include "motis/bootstrap/motis_instance.h"

#include <algorithm>
#include <exception>
#include <iostream>

#include "boost/filesystem.hpp"

#include "utl/pipes.h"
#include "utl/progress_tracker.h"
#include "utl/raii.h"
#include "utl/verify.h"

#include "motis/core/common/logging.h"
#include "motis/core/schedule/schedule_data_key.h"
#include "motis/module/context/motis_call.h"
#include "motis/module/context/motis_publish.h"
#include "motis/bootstrap/import_files.h"
#include "motis/bootstrap/import_schedule.h"
#include "motis/loader/loader.h"

#include "modules.h"

using namespace motis::module;
using namespace motis::logging;
namespace fs = boost::filesystem;

namespace motis::bootstrap {

motis_instance::motis_instance() : controller{build_modules()} {
  for (auto& m : modules_) {
    m->set_shared_data(&shared_data_);
  }
}

void motis_instance::stop_remotes() {
  for (auto const& r : remotes_) {
    r->stop();
  }
  remotes_.clear();
}

std::vector<motis::module::module*> motis_instance::modules() const {
  std::vector<motis::module::module*> m;
  for (auto& module : modules_) {
    m.push_back(module.get());
  }
  return m;
}

std::vector<std::string> motis_instance::module_names() const {
  std::vector<std::string> s;
  for (auto const& module : modules_) {
    s.push_back(module->module_name());
  }
  return s;
}

schedule const& motis_instance::sched() const {
  return *shared_data_.get<schedule_data>(SCHEDULE_DATA_KEY).schedule_;
}

void motis_instance::import(module_settings const& module_opt,
                            loader::loader_options const& dataset_opt,
                            import_settings const& import_opt,
                            bool const silent) {
  auto bars = utl::global_progress_bars{silent};

  registry_.subscribe("/import", import_files);
  register_import_schedule(*this, dataset_opt, import_opt.data_directory_);

  for (auto const& module : modules_) {
    if (module_opt.is_module_active(module->module_name())) {
      module->set_data_directory(import_opt.data_directory_);
      module->import(registry_);
    }
  }

  // Dummy message to trigger initial progress updates.
  publish(make_success_msg("/import"), 1);

  message_creator fbb;
  std::vector<flatbuffers::Offset<flatbuffers::String>> import_paths;
  for (auto const& path : import_opt.import_paths_) {
    if (!fs::exists(path)) {
      LOG(warn) << "file does not exist, skipping: " << path;
      continue;
    }
    import_paths.push_back(fbb.CreateString(path));
  }
  fbb.create_and_finish(
      MsgContent_FileEvent,
      motis::import::CreateFileEvent(fbb, fbb.CreateVector(import_paths))
          .Union(),
      "/import", DestinationType_Topic);
  publish(make_msg(fbb), 1);

  registry_.reset();

  utl::verify(shared_data_.includes(SCHEDULE_DATA_KEY), "schedule not loaded");

  if (import_opt.require_successful_) {
    auto const unsuccessful_imports =
        utl::all(modules_)  //
        | utl::remove_if([&](auto&& m) {
            return !module_opt.is_module_active(m->module_name()) ||
                   m->import_successful();
          })  //
        | utl::transform([&](auto&& m) { return m->module_name(); })  //
        | utl::vec();
    utl::verify(unsuccessful_imports.empty(),
                "some imports were not successful: {}", unsuccessful_imports);
  }
}

void motis_instance::init_modules(module_settings const& module_opt,
                                  unsigned num_threads) {
  for (auto const& module : modules_) {
    if (!module_opt.is_module_active(module->prefix())) {
      continue;
    }

    if (!module->import_successful()) {
      LOG(info) << module->module_name() << ": import was not successful";
      continue;
    }

    try {
      module->init(registry_);
    } catch (std::exception const& e) {
      LOG(emrg) << "module " << module->module_name()
                << ": unhandled init error: " << e.what();
      throw;
    } catch (...) {
      LOG(emrg) << "module " << module->module_name()
                << "unhandled unknown init error";
      throw;
    }
  }
  publish("/init", num_threads);
}

void motis_instance::init_remotes(
    std::vector<std::pair<std::string, std::string>> const& remotes) {
  for (auto const& [host, port] : remotes) {
    remotes_
        .emplace_back(std::make_unique<remote>(
            *this, runner_.ios(), host, port,
            [&]() {
              ++connected_remotes_;
              if (connected_remotes_ == remotes_.size()) {
                retry_no_target_msgs();
                if (on_remotes_registered_) {
                  on_remotes_registered_();
                  on_remotes_registered_ = nullptr;
                }
              }
            },
            [&]() { --connected_remotes_; }))
        ->start();
  }
}

msg_ptr motis_instance::call(std::string const& target, unsigned num_threads) {
  return call(make_no_msg(target), num_threads);
}

msg_ptr motis_instance::call(msg_ptr const& msg, unsigned num_threads) {
  std::exception_ptr e;
  msg_ptr response;

  run(
      [&]() {
        try {
          response = motis_call(msg)->val();
        } catch (...) {
          e = std::current_exception();
        }
      },
      access_of(msg), num_threads);

  if (e) {
    std::rethrow_exception(e);
  }

  return response;
}

void motis_instance::publish(std::string const& target, unsigned num_threads) {
  publish(make_no_msg(target), num_threads);
}

void motis_instance::publish(msg_ptr const& msg, unsigned num_threads) {
  std::exception_ptr e;

  run(
      [&]() {
        try {
          ctx::await_all(motis_publish(msg));
        } catch (...) {
          e = std::current_exception();
        }
      },
      access_of(msg), num_threads);

  if (e) {
    std::rethrow_exception(e);
  }
}

}  // namespace motis::bootstrap
