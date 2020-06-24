#pragma once

#include <string>
#include <vector>

#include "boost/asio/io_service.hpp"
#include "boost/asio/strand.hpp"
#include "boost/filesystem/path.hpp"

#include "conf/configuration.h"

#include "motis/module/message.h"
#include "motis/module/registry.h"
#include "motis/module/shared_data.h"

namespace motis {

struct schedule;

namespace module {

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
  void set_shared_data(shared_data*);

  virtual void import(registry&) {}
  virtual void init(registry&) {}

  virtual bool import_successful() const { return true; }

protected:
  schedule const& get_sched() const;

  template <typename T>
  T const& get_shared_data(std::string_view const s) const {
    return shared_data_->get<T>(s);
  }

  template <typename T>
  T& get_shared_data_mutable(std::string_view const s) {
    return shared_data_->get<T>(s);
  }

  template <typename T>
  T const* find_shared_data(std::string_view const s) const {
    return shared_data_->find<T>(s);
  }

  template <typename T>
  void add_shared_data(std::string_view const s, T&& data) {
    shared_data_->emplace_data(s, std::forward<T>(data));
  }

  boost::filesystem::path const& get_data_directory() const;

private:
  boost::filesystem::path data_directory_;
  shared_data* shared_data_{nullptr};
};

}  // namespace module
}  // namespace motis
