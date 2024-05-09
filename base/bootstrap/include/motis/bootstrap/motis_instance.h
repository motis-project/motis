#pragma once

#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "boost/asio/io_service.hpp"

#include "motis/module/controller.h"
#include "motis/module/message.h"
#include "motis/module/module.h"
#include "motis/module/remote.h"
#include "motis/bootstrap/import_settings.h"
#include "motis/bootstrap/module_settings.h"

namespace motis::bootstrap {

struct motis_instance : public motis::module::controller {
  motis_instance();

  motis_instance(motis_instance const&) = delete;
  motis_instance& operator=(motis_instance const&) = delete;

  motis_instance(motis_instance&&) = delete;
  motis_instance& operator=(motis_instance&&) = delete;

  ~motis_instance() override = default;

  void stop_remotes();

  void stop_io();
  void init_io(module_settings const&);

  void on_remotes_registered(std::function<void()> fn) {
    on_remotes_registered_ = std::move(fn);
  }

  std::vector<module::module*> modules() const;
  std::vector<std::string> module_names() const;
  schedule const& sched() const;

  void import(module_settings const&, import_settings const&,
              bool silent = false);
  void init_modules(module_settings const&,
                    unsigned num_threads = std::thread::hardware_concurrency());
  void init_remotes(
      std::vector<std::pair<std::string, std::string>> const& remotes);

  module::msg_ptr call(
      std::string const& target,
      unsigned num_threads = std::thread::hardware_concurrency(),
      std::vector<ctx::access_request>&& access = {});
  module::msg_ptr call(
      module::msg_ptr const&,
      unsigned num_threads = std::thread::hardware_concurrency(),
      std::vector<ctx::access_request>&& access = {});

  void publish(std::string const& target,
               unsigned num_threads = std::thread::hardware_concurrency(),
               std::vector<ctx::access_request>&& access = {});
  void publish(module::msg_ptr const&,
               unsigned num_threads = std::thread::hardware_concurrency(),
               std::vector<ctx::access_request>&& access = {});

  std::vector<std::shared_ptr<motis::module::remote>> remotes_;
  std::function<void()> on_remotes_registered_;
  unsigned connected_remotes_{0};
};

}  // namespace motis::bootstrap
