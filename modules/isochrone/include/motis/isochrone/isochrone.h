#pragma once

#include <memory>
#include <mutex>
#include <vector>

#include "motis/module/module.h"

namespace motis::isochrone {

struct memory;

struct isochrone : public motis::module::module {
  isochrone();
  void init(motis::module::registry&) override;

private:
  motis::module::msg_ptr list_stations(const motis::module::msg_ptr& msg);

  std::mutex mem_pool_mutex_;
  std::vector<std::unique_ptr<memory>> mem_pool_;
};

}  // namespace motis::isochrone
