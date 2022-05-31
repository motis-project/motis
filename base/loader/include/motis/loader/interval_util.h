#pragma once

#include <cinttypes>
#include <utility>

namespace motis {

struct schedule;

namespace loader {

struct Interval;  // NOLINT

std::pair<int, int> first_last_days(schedule const& sched,
                                    std::size_t src_index, Interval const*);

}  // namespace loader
}  // namespace motis
