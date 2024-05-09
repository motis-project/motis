#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "boost/asio/io_service.hpp"
#include "boost/asio/strand.hpp"

#include "ctx/ctx.h"

#include "conf/configuration.h"

#include "motis/module/ctx_data.h"
#include "motis/module/dispatcher.h"
#include "motis/module/import_dispatcher.h"
#include "motis/module/locked_resources.h"
#include "motis/module/message.h"
#include "motis/module/registry.h"
#include "motis/module/subc_reg.h"

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

  virtual void reg_subc(subc_reg&) {}

  std::string const& module_name() const { return prefix(); }

  std::string data_path(std::filesystem::path const&) const;
  void set_data_directory(std::string const&);
  void set_shared_data(dispatcher*);

  virtual void import(import_dispatcher&) {}
  virtual void init(registry&) {}

  virtual bool import_successful() const { return true; }

  virtual void init_io(boost::asio::io_context&) {}
  virtual void stop_io() {}

  template <typename T>
  T const& get_shared_data(ctx::res_id_t const id) const {
    return shared_data_->get<T>(id);
  }

  template <typename T>
  T& get_shared_data_mutable(ctx::res_id_t const id) {
    return shared_data_->get<T>(id);
  }

  template <typename T>
  T const* find_shared_data(ctx::res_id_t const id) const {
    return shared_data_->find<T>(id);
  }

  template <typename T>
  void add_shared_data(ctx::res_id_t const id, T&& data) {
    shared_data_->emplace_data(id, std::forward<T>(data));
  }

  void remove_shared_data(ctx::res_id_t const id) { shared_data_->remove(id); }

  ctx::res_id_t generate_res_id() { return shared_data_->generate_res_id(); }

  locked_resources lock_resources(
      ctx::accesses_t access, ctx::op_type_t op_type = ctx::op_type_t::WORK);

  std::filesystem::path const& get_data_directory() const;

  dispatcher* shared_data_{nullptr};
  std::filesystem::path data_directory_;
};

}  // namespace module
}  // namespace motis
