#pragma once

#include "motis/core/schedule/schedule.h"

namespace motis {

enum schedule_access { RO, RW };

template <schedule_access A>
struct synced_schedule {};

template <>
struct synced_schedule<RO> {
  explicit synced_schedule(schedule& s) : s_(s) {}
  schedule const& sched() const { return s_; }
  schedule const& s_;
};

template <>
struct synced_schedule<RW> {
  explicit synced_schedule(schedule& s) : s_(s) {}
  schedule& sched() { return s_; }
  schedule& s_;
};

}  // namespace motis
