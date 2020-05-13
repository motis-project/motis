#pragma once

#include <string>
#include <vector>

#include "boost/asio/io_service.hpp"
#include "boost/asio/strand.hpp"
#include "boost/filesystem/path.hpp"

#include "conf/configuration.h"

#include "motis/core/schedule/synced_schedule.h"
#include "motis/module/message.h"
#include "motis/module/progress_listener.h"
#include "motis/module/registry.h"

namespace motis::module {

struct module : public conf::configuration {
  explicit module(std::string name = "", std::string prefix = "")
      : configuration(std::move(name), std::move(prefix)) {}

  module(module const&) = delete;
  module& operator=(module const&) = delete;

  module(module&&) = delete;
  module& operator=(module&&) = delete;

  ~module() override = default;

  std::string const& module_name() const { return prefix(); }

  std::string data_path(boost::filesystem::path const&);
  void set_data_directory(std::string const&);
  void set_context(motis::schedule& schedule);

  virtual void import(progress_listener&, registry&) {}
  virtual void init(registry&) {}

  virtual bool import_successful() const { return true; }

protected:
  template <schedule_access A>
  synced_schedule<A> synced_sched() {
    return synced_schedule<A>(*schedule_);
  }

  boost::filesystem::path const& get_data_directory() const;

private:
  boost::filesystem::path data_directory_;
  motis::schedule* schedule_ = nullptr;
};

}  // namespace motis::module
