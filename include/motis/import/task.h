#pragma once

#include <filesystem>
#include <functional>
#include <initializer_list>
#include <string>
#include <string_view>
#include <vector>

#include "dataset_hashes.h"
#include "motis/fwd.h"
#include "motis/hashes.h"

namespace motis {

struct task {
  task(std::string_view name,
       std::filesystem::path const& data_path,
       config const&,
       meta_t);
  virtual ~task();

  virtual void run() = 0;
  virtual bool is_enabled() const = 0;

  void add_dependency(std::initializer_list<std::reference_wrapper<task>>);
  void exec();
  std::string_view name() const;
  bool is_done() const;
  bool is_ready_to_run() const;
  bool can_load() const;

  meta_t hashes_;
  std::filesystem::path const& data_path_;
  config const& c_;

private:
  std::string name_;
  bool done_{false};
  std::vector<task const*> out_, in_;
};

}  // namespace motis
