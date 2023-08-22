#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "cista/reflection/comparable.h"

#include "motis/core/schedule/schedule.h"

#include "motis/paxmon/compact_journey.h"

namespace motis::paxmon {

struct service_info {
  CISTA_COMPARABLE();

  std::string name_;
  std::string_view category_;
  std::uint32_t train_nr_{};
  std::string_view line_;
  std::string_view provider_;
  service_class clasz_{service_class::OTHER};
};

service_info get_service_info(schedule const& sched, connection_info const* ci,
                              service_class const clasz);

std::vector<std::pair<service_info, unsigned>> get_service_infos(
    schedule const& sched, trip const* trp);

std::vector<std::pair<service_info, unsigned>> get_service_infos_for_leg(
    schedule const& sched, journey_leg const& leg);

}  // namespace motis::paxmon
