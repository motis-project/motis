#pragma once

#include <memory>

#include "motis/module/message.h"

namespace motis::osrm {

struct router {
  explicit router(std::string const& path);
  ~router();

  router(router const&) = delete;
  router& operator=(router const&) = delete;

  router(router&&) = delete;
  router& operator=(router&&) = delete;

  motis::module::msg_ptr table(OSRMManyToManyRequest const*) const;
  motis::module::msg_ptr one_to_many(OSRMOneToManyRequest const*) const;
  motis::module::msg_ptr via(OSRMViaRouteRequest const*) const;
  motis::module::msg_ptr smooth_via(OSRMSmoothViaRouteRequest const*) const;

  struct impl;
  std::unique_ptr<impl> impl_;
};

}  // namespace motis::osrm
