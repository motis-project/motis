#pragma once

#include <filesystem>
#include <vector>

#include "motis/fwd.h"

namespace motis {

struct task {
  task(std::filesystem::path const& data_path, data&, config&);
  virtual ~task();

  virtual void load() = 0;
  virtual void unload() = 0;
  virtual void run() = 0;
  virtual bool is_enabled() const = 0;

  void exec();
  std::string_view name() const;
  bool is_done() const;
  bool is_ready_to_run() const;
  bool can_load() const;

  std::filesystem::path const& data_path_;
  config& c_;
  data& d_;

private:
  std::string name_;
  bool done_{false};
  std::vector<task const*> out_, in_;
};

}  // namespace motis