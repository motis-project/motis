#include "motis/bootstrap/motis_instance.h"

#include <algorithm>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>

#include "fmt/format.h"
#include "fmt/ranges.h"

#include "prometheus/registry.h"

#include "opentelemetry/trace/tracer.h"

#include "utl/pipes.h"
#include "utl/progress_tracker.h"
#include "utl/raii.h"
#include "utl/verify.h"

#include "motis/core/common/logging.h"
#include "motis/core/otel/tracer.h"
#include "motis/module/context/motis_call.h"
#include "motis/module/context/motis_publish.h"
#include "motis/bootstrap/import_files.h"

#include "modules.h"

using namespace motis::module;
using namespace motis::logging;

namespace motis::bootstrap {

motis_instance::motis_instance() : controller{build_modules()} {
  emplace_data(to_res_id(global_res_id::METRICS),
               std::make_shared<prometheus::Registry>());
  for (auto& m : modules_) {
    m->set_shared_data(this);
  }
}

void motis_instance::stop_remotes() {
  for (auto const& r : remotes_) {
    r->stop();
  }
  remotes_.clear();
}

void motis_instance::stop_io() {
  for (auto const& module : modules_) {
    module->stop_io();
  }
}

std::vector<motis::module::module*> motis_instance::modules() const {
  std::vector<motis::module::module*> m;
  m.reserve(modules_.size());
  for (auto& module : modules_) {
    m.push_back(module.get());
  }
  return m;
}

std::vector<std::string> motis_instance::module_names() const {
  std::vector<std::string> s;
  s.reserve(modules_.size());
  for (auto const& module : modules_) {
    s.push_back(module->module_name());
  }
  return s;
}

void motis_instance::import(module_settings const& module_opt,
                            import_settings const& import_opt,
                            bool const silent) {
  auto bars = utl::global_progress_bars{silent};

  auto span = motis_tracer->StartSpan("import");
  auto scope = opentelemetry::trace::Scope{span};

  auto dispatcher = import_dispatcher{};

  register_import_files(dispatcher);

  for (auto const& module : modules_) {
    if (module_opt.is_module_active(module->module_name())) {
      module->set_data_directory(import_opt.data_directory_);
      module->import(dispatcher);
    }
  }

  // Dummy message to trigger initial progress updates.
  dispatcher.publish(make_success_msg("/import"));
  dispatcher.run();

  // Paths as actual trigger for import processing.
  dispatcher.publish(make_file_event(import_opt.import_paths_));
  dispatcher.run();

  registry_.reset();

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
  auto outer_span = motis_tracer->StartSpan("init_modules");
  auto outer_scope = opentelemetry::trace::Scope{outer_span};

  for (auto const& module : modules_) {
    if (!module_opt.is_module_active(module->prefix())) {
      continue;
    }

    auto span =
        motis_tracer->StartSpan(fmt::format("init {}", module->module_name()));
    auto scope = opentelemetry::trace::Scope{outer_span};

    if (!module->import_successful()) {
      LOG(info) << module->module_name() << ": import was not successful";
      span->SetStatus(opentelemetry::trace::StatusCode::kError,
                      "import failed");
      continue;
    }

    try {
      module->init(registry_);
    } catch (std::exception const& e) {
      span->AddEvent("exception", {
                                      {"exception.message", e.what()},
                                  });
      span->SetStatus(opentelemetry::trace::StatusCode::kError, "exception");
      LOG(emrg) << "module " << module->module_name()
                << ": unhandled init error: " << e.what();
      throw;
    } catch (...) {
      span->AddEvent("exception", {{"exception.type", "unknown"}});
      span->SetStatus(opentelemetry::trace::StatusCode::kError,
                      "unknown error");
      LOG(emrg) << "module " << module->module_name()
                << "unhandled unknown init error";
      throw;
    }
  }
  publish("/init", num_threads);
}

void motis_instance::init_io(module_settings const& module_opt) {
  for (auto const& module : modules_) {
    if (!module_opt.is_module_active(module->prefix())) {
      continue;
    }
    module->init_io(runner_.ios());
  }

  for (auto const& [name, t] : timers_) {
    t->exec(boost::system::error_code{});
  }
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

msg_ptr motis_instance::call(std::string const& target, unsigned num_threads,
                             std::vector<ctx::access_request>&& access) {
  return call(make_no_msg(target), num_threads, std::move(access));
}

msg_ptr motis_instance::call(msg_ptr const& msg, unsigned num_threads,
                             std::vector<ctx::access_request>&& access) {
  if (direct_mode_dispatcher_ != nullptr) {
    ctx_data const data{dispatcher::direct_mode_dispatcher_};
    return static_cast<dispatcher*>(this)->req(msg, data, ctx::op_id{})->val();
  } else {
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
        std::move(access), num_threads);

    if (e) {
      std::rethrow_exception(e);
    }

    return response;
  }
}

void motis_instance::publish(std::string const& target, unsigned num_threads,
                             std::vector<ctx::access_request>&& access) {
  publish(make_no_msg(target), num_threads, std::move(access));
}

void motis_instance::publish(msg_ptr const& msg, unsigned num_threads,
                             std::vector<ctx::access_request>&& access) {
  if (direct_mode_dispatcher_ != nullptr) {
    ctx_data const data{dispatcher::direct_mode_dispatcher_};
    static_cast<dispatcher*>(this)->publish(msg, data, ctx::op_id{});
  } else {
    std::exception_ptr e;

    run(
        [&]() {
          try {
            ctx::await_all(motis_publish(msg));
          } catch (...) {
            e = std::current_exception();
          }
        },
        std::move(access), num_threads);

    if (e) {
      std::rethrow_exception(e);
    }
  }
}

}  // namespace motis::bootstrap
