#pragma once

#include <utility>

namespace motis {

struct schedule;

namespace loader {

struct Interval;  // NOLINT

std::pair<int, int> first_last_days(schedule const& sched, Interval const*);

}  // namespace loader
}  // namespace motis
