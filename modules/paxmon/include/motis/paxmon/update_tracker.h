#pragma once

#include <memory>
#include <utility>

#include "motis/core/schedule/schedule.h"
#include "motis/module/message.h"

#include "motis/paxmon/passenger_group.h"

namespace motis::paxmon {

struct universe;

struct update_tracker {
  update_tracker();
  update_tracker(update_tracker const&);
  update_tracker(update_tracker&&) noexcept;
  ~update_tracker();

  update_tracker& operator=(update_tracker const&);
  update_tracker& operator=(update_tracker&&) noexcept;

  void start_tracking(universe const&, schedule const&,
                      bool include_before_trip_load_info,
                      bool include_after_trip_load_info);
  std::pair<motis::module::message_creator&,
            flatbuffers::Offset<PaxMonTrackedUpdates>>
  finish_updates();
  void stop_tracking();
  bool is_tracking() const;

  void before_group_added(passenger_group const*);
  void before_group_reused(passenger_group const*);
  void after_group_reused(passenger_group const*);
  void before_group_removed(passenger_group const*);
  void before_trip_rerouted(trip const*);

private:
  struct impl;
  std::unique_ptr<impl> impl_;
};

}  // namespace motis::paxmon
