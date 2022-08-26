#pragma once

#include <memory>
#include <string>
#include <vector>

#include "motis/module/module.h"

namespace motis::nigiri {

struct config {
  std::vector<std::string> timetables_;
  bool cache_timetable_{true};
};

struct nigiri : public motis::module::module {
  nigiri();
  ~nigiri() override;

  nigiri(nigiri const&) = delete;
  nigiri& operator=(nigiri const&) = delete;

  nigiri(nigiri&&) = delete;
  nigiri& operator=(nigiri&&) = delete;

  void init(motis::module::registry&) override;
  void import(motis::module::import_dispatcher&) override;
  bool import_successful() const override { return import_successful_; }

private:
  bool import_successful_{false};

  struct impl;
  std::unique_ptr<impl> impl_;
  config config_;
};

}  // namespace motis::nigiri
