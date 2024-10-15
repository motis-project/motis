#include "motis/revise/update_journey_status.h"

#include <algorithm>
#include <numeric>

#include "utl/verify.h"

#include "motis/core/common/interval_map.h"
#include "motis/core/common/unixtime.h"
#include "motis/core/access/station_access.h"
#include "motis/core/access/track_access.h"

#include "motis/revise/get_interchanges.h"

namespace motis::revise {

int get_walk_time(journey const& j, std::string const& station_id,
                  unixtime const schedule_time) {
  // Get stop index.
  auto const stop_it =
      std::find_if(begin(j.stops_), end(j.stops_), [&](journey::stop const& s) {
        return s.departure_.schedule_timestamp_ == schedule_time &&
               s.eva_no_ == station_id;
      });
  utl::verify(stop_it != end(j.stops_),
              "get walk time(stop) : invalid journey");
  auto const stop_idx = std::distance(begin(j.stops_), stop_it);

  // Find first walk.
  auto const from_it = std::find_if(begin(j.transports_), end(j.transports_),
                                    [&](journey::transport const& t) {
                                      return t.is_walk_ && t.from_ == stop_idx;
                                    });
  utl::verify(from_it != end(j.transports_),
              "get walk stop(first walk) : invalid journey");
  // Find last walk.
  auto const to_it =
      std::find_if(from_it, end(j.transports_),
                   [](journey::transport const& t) { return !t.is_walk_; });
  if (to_it != end(j.transports_)) {
    // Sum up walks [first walk, last walk].
    return std::accumulate(from_it, to_it, 0,
                           [](unsigned sum, journey::transport const& t) {
                             return sum + (t.duration_ * 60);
                           });
  } else {
    return 0;
  }
}

bool is_same_platform(schedule const& sched, station const* st,
                      std::string const& track1_name,
                      std::string const& track2_name) {
  auto const track1 = get_track_index(sched, track1_name);
  auto const track2 = get_track_index(sched, track2_name);
  if (track1.has_value() && track2.has_value()) {
    auto const platform1 = st->get_platform(track1.value());
    auto const platform2 = st->get_platform(track2.value());
    return platform1.has_value() && platform1 == platform2;
  } else {
    return false;
  }
}

int get_transfer_time(schedule const& sched, journey const& j,
                      extern_interchange const& ic) {
  if (get_station(sched, ic.first_stop_.eva_no_)->index_ ==
      get_station(sched, ic.second_stop_.eva_no_)->index_) {
    auto const station =
        sched.stations_[get_station(sched, ic.first_stop_.eva_no_)->index_]
            .get();
    return is_same_platform(sched, station, ic.first_stop_.arrival_.track_,
                            ic.second_stop_.departure_.track_)
               ? station->platform_transfer_time_ * 60
               : station->transfer_time_ * 60;
  } else {
    return get_walk_time(j, ic.first_stop_.eva_no_,
                         ic.first_stop_.departure_.schedule_timestamp_);
  }
}

void update_interchange_status(schedule const& sched, journey& j) {
  auto const interchanges = get_interchanges(j);
  for (auto const& ic : interchanges) {
    auto const transfer_time = get_transfer_time(sched, j, ic);
    if (ic.second_stop_.departure_.timestamp_ -
            ic.first_stop_.arrival_.timestamp_ <
        transfer_time) {
      j.problems_.emplace_back(
          journey::problem{journey::problem_type::INTERCHANGE_TIME_VIOLATED,
                           static_cast<unsigned>(ic.first_stop_idx_),
                           static_cast<unsigned>(ic.second_stop_idx_)});
    }
  }
}

void update_canceled_train_status(journey& j) {
  interval_map<int> canceled_trains;
  unsigned counter = 0;
  for (auto it = begin(j.stops_); it != end(j.stops_); ++it) {
    auto const arrival_valid = it->arrival_.valid_ || it == begin(j.stops_);
    auto const departure_valid =
        it->departure_.valid_ || std::next(it) == end(j.stops_);
    if (arrival_valid && departure_valid) {
      continue;
    }
    canceled_trains.add_entry(counter, std::distance(begin(j.stops_), it));
  }

  for (auto const& ranges : canceled_trains.get_attribute_ranges()) {
    for (auto const& range : ranges.second) {
      j.problems_.emplace_back(
          journey::problem{journey::problem_type::CANCELED_TRAIN,
                           static_cast<unsigned>(range.from_),
                           static_cast<unsigned>(range.to_)});
    }
  }
}

void update_journey_status(schedule const& sched, journey& j) {
  update_interchange_status(sched, j);
  update_canceled_train_status(j);
}

}  // namespace motis::revise
