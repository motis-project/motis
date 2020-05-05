#include "motis/bootstrap/motis_instance.h"

#include <chrono>
#include <algorithm>
#include <atomic>
#include <exception>
#include <future>
#include <iostream>
#include <thread>

#if defined(_MSC_VER)
#include <io.h>
#include <windows.h>
#endif

#include "boost/filesystem.hpp"

#include "cista/hash.h"
#include "cista/mmap.h"

#include "ctx/future.h"

#include "motis/core/common/logging.h"
#include "motis/module/context/motis_call.h"
#include "motis/module/context/motis_publish.h"
#include "motis/loader/loader.h"

#include "modules.h"

using namespace motis::module;
using namespace motis::logging;

namespace motis::bootstrap {

#ifdef _MSC_VER

void move(int x, int y) {
  auto hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
  if (!hStdout) return;

  CONSOLE_SCREEN_BUFFER_INFO csbiInfo;
  GetConsoleScreenBufferInfo(hStdout, &csbiInfo);

  COORD cursor;

  cursor.X = csbiInfo.dwCursorPosition.X + x;
  cursor.Y = csbiInfo.dwCursorPosition.Y + y;
  SetConsoleCursorPosition(hStdout, cursor);
}

void move_cursor_up(int lines) {
  if (lines != 0) {
    move(0, -lines);
  }
}

#else

void move_cursor_up(int lines) {
  if (lines != 0) {
    std::cout << "\x1b[" << lines << "A";
  }
}

#endif

void clear_line() {
  auto const empty =
      "          "
      "          "
      "          "
      "          "
      "          "
      "          "
      "          "
      "          "
      "          "
      "          ";
  std::cout << empty << "\r";
}

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
                            std::vector<std::string> const& import_paths,
                            std::string const& data_directory) {
  registry_.subscribe("/import", [&](msg_ptr const& msg) -> msg_ptr {
    if (msg->get()->content_type() != MsgContent_FileEvent) {
      return nullptr;
    }

    using motis::import::FileEvent;
    auto const path = motis_content(FileEvent, msg)->path()->str();
    auto const name = boost::filesystem::path{path}.filename().generic_string();
    if (name.substr(name.size() - 8) != ".osm.pbf") {
      return nullptr;
    }

    cista::mmap m(path.c_str(), cista::mmap::protection::READ);
    auto const hash = cista::hash(std::string_view{
        reinterpret_cast<char const*>(m.begin()),
        std::max(static_cast<size_t>(50 * 1024 * 1024), m.size())});

    message_creator fbb;
    fbb.create_and_finish(MsgContent_OSMEvent,
                          motis::import::CreateOSMEvent(
                              fbb, fbb.CreateString(path), hash, m.size())
                              .Union(),
                          "/import", DestinationType_Topic);
    ctx::await_all(motis_publish(make_msg(fbb)));
    return nullptr;
  });

  struct state {
    std::vector<std::string> dependencies_;
    import::Status status_{import::Status::Status_WAITING};
    int progress_{0U};
  };
  std::map<std::string, state> status;
  auto last_print_height = 0U;
  auto const print_status = [&]() {
    move_cursor_up(last_print_height);
    for (auto const& [name, s] : status) {
      clear_line();
      std::cout << std::setw(20) << std::setfill(' ') << std::right << name
                << ": ";
      switch (s.status_) {
        case import::Status_WAITING:
          std::cout << "WAITING, dependencies: ";
          for (auto const& dep : s.dependencies_) {
            std::cout << dep << " ";
          }
          break;
        case import::Status_FINISHED: std::cout << "DONE"; break;
        case import::Status_RUNNING:
          std::cout << "[";
          constexpr auto const WIDTH = 55U;
          bool end = false;
          for (auto i = 0U; i < 55U; ++i) {
            auto const scaled = static_cast<int>(i * 100.0 / WIDTH);
            std::cout << (scaled < s.progress_
                              ? '='
                              : !end && scaled >= s.progress_ ? '>' : ' ');
            end = scaled >= s.progress_;
          }
          std::cout << "] " << s.progress_ << "%";
          break;
      }
      std::cout << "\n";
    }
    last_print_height = status.size();
  };

  registry_.subscribe("/import", [&](msg_ptr const& msg) -> msg_ptr {
    if (msg->get()->content_type() != MsgContent_StatusUpdate) {
      return nullptr;
    }

    using import::StatusUpdate;
    auto const upd = motis_content(StatusUpdate, msg);

    status[upd->name()->str()] = {
        utl::to_vec(*upd->waiting_for(), [](auto&& e) { return e->str(); }),
        upd->status(), upd->progress()};

    print_status();

    return nullptr;
  });

  std::cout << "\nImport:\n\n";
  for (auto const& module : modules_) {
    if (is_module_active(modules, exclude_modules, module->name())) {
      module->set_data_directory(data_directory);
      module->import(registry_);
    }
  }

  // Dummy message asking for initial dependencies of every job.
  publish(make_success_msg("/import"), 1);

  logging::log::enabled_ = false;
  for (auto const& path : import_paths) {
    message_creator fbb;
    fbb.create_and_finish(
        MsgContent_FileEvent,
        motis::import::CreateFileEvent(fbb, fbb.CreateString(path)).Union(),
        "/import", DestinationType_Topic);
    publish(make_msg(fbb), 1);
  }
  logging::log::enabled_ = true;

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
