#pragma once

#include <vector>

#include "motis/core/schedule/bitfield.h"
#include "motis/core/schedule/connection.h"
#include "motis/core/schedule/schedule.h"
#include "motis/core/schedule/time.h"

namespace motis::loader {

struct route_t {
  route_t();

  route_t(std::vector<light_connection> const& new_lcons,
          std::vector<time> const& times, schedule const& sched);

  bool add_service(mcd::vector<light_connection> const& new_lcons,
                   std::vector<time> const& new_times, schedule const& sched);

  void verify_sorted();

  bool empty() const;

  void update_traffic_days(std::vector<light_connection> const& new_lcons,
                           schedule const&);

  std::vector<light_connection> const& operator[](size_t) const;

  std::vector<std::vector<time>> times_;
  std::vector<std::vector<light_connection>> lcons_;
  bitfield traffic_days_;
};

}  // namespace motis::loader
