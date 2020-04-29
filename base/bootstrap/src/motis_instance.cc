#include "motis/bootstrap/motis_instance.h"

#include <chrono>
#include <algorithm>
#include <atomic>
#include <exception>
#include <future>
#include <thread>

#include "ctx/future.h"

#include "motis/core/common/logging.h"
#include "motis/module/context/motis_call.h"
#include "motis/module/context/motis_publish.h"
#include "motis/loader/loader.h"

#include "modules.h"

using namespace motis::module;
using namespace motis::logging;

namespace motis::bootstrap {

bool is_module_active(std::vector<std::string> const& yes,
                      std::vector<std::string> const& no,
                      std::string const& module) {
  return std::find(begin(yes), end(yes), module) != end(yes) &&
         std::find(begin(no), end(no), module) == end(no);
}

motis_instance::motis_instance() : controller(build_modules()) {}

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
    s.push_back(module->name());
  }
  return s;
}

void motis_instance::init_schedule(
    motis::loader::loader_options const& dataset_opt) {
  schedule_ = loader::load_schedule(dataset_opt, schedule_buf_);
  sched_ = schedule_.get();
}

void motis_instance::import(std::vector<std::string> const& modules,
                            std::vector<std::string> const& exclude_modules,
                            std::vector<std::string> const& import_paths) {
  for (auto const& module : modules_) {
    if (is_module_active(modules, exclude_modules, module->name())) {
      module->import(registry_);
    }
  }

  for (auto const& path : import_paths) {
    message_creator fbb;
    fbb.create_and_finish(
        MsgContent_FileEvent,
        motis::import::CreateFileEvent(fbb, fbb.CreateString(path)).Union(),
        "/import", DestinationType_Topic);
    publish(make_msg(fbb), 1);
  }

  registry_.reset();
}

void motis_instance::init_modules(
    std::vector<std::string> const& modules,
    std::vector<std::string> const& exclude_modules, unsigned num_threads) {
  for (auto const& module : modules_) {
    if (!is_module_active(modules, exclude_modules, module->name())) {
      continue;
    }

    try {
      module->set_context(*schedule_);
      module->init(registry_);
    } catch (std::exception const& e) {
      LOG(emrg) << "module " << module->name()
                << ": unhandled init error: " << e.what();
      throw;
    } catch (...) {
      LOG(emrg) << "module " << module->name()
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
