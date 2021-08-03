#pragma once

#include <ctime>
#include <optional>

#include "flatbuffers/flatbuffers.h"

#include "motis/hash_map.h"

#include "motis/core/common/unixtime.h"
#include "motis/core/schedule/event_type.h"
#include "motis/core/schedule/time.h"
#include "motis/core/schedule/trip.h"
#include "motis/core/access/time_access.h"

#include "gtfsrt.pb.h"

namespace motis {
struct schedule;

namespace ris::gtfsrt {

struct stop_context;

struct evt {
  evt() = default;
  evt(trip const&, stop_context const&, event_type);
  evt(evt const& o) = default;
  evt& operator=(evt const&) = default;
  evt(evt&&) = default;
  evt& operator=(evt&&) = default;
  ~evt() = default;

  inline void verify_times(schedule& sched) const {
    verify_timestamp(sched, orig_sched_time_);
    if (new_sched_time_ > 0) {
      verify_timestamp(sched, new_sched_time_);
    }
  }

  int stop_idx_{std::numeric_limits<int>::max()};
  int seq_no_{std::numeric_limits<int>::max()};
  std::string stop_id_;
  unixtime orig_sched_time_{0};
  unixtime new_sched_time_{0};
  uint64_t train_nr_{0};
  std::string line_id_;
  event_type type_{event_type::ARR};
};

struct message_context {
  explicit message_context(unixtime const timestamp) : timestamp_{timestamp} {};

  inline void adjust_times(const unixtime time) {
    earliest_ = std::min(earliest_, time);
    latest_ = std::max(latest_, time);
  }

  flatbuffers::FlatBufferBuilder b_;
  unixtime timestamp_, earliest_{std::numeric_limits<unixtime>::max()},
      latest_{std::numeric_limits<unixtime>::min()};
};

struct known_stop_skips {
  known_stop_skips(std::string trip_id, unixtime time)
      : trip_id(std::move(trip_id)), trip_date{time} {}

  inline bool is_skipped(int const seq_no) {
    auto it = skipped_stops_.find(seq_no);
    return it != end(skipped_stops_) ? it->second : false;
  }

  std::string const trip_id;
  unixtime const trip_date;
  mcd::hash_map<int /*seq-no */, bool> skipped_stops_;
};

struct known_addition_trip {
  friend bool operator<(known_addition_trip const& lhs,
                        known_addition_trip const& rhs) {
    return lhs.gtfs_id_ < rhs.gtfs_id_;
  }

  friend bool operator==(known_addition_trip const& lhs,
                         known_addition_trip const& rhs) {
    return lhs.gtfs_id_ == rhs.gtfs_id_;
  }

  known_addition_trip() = default;
  explicit known_addition_trip(gtfs_trip_id gtfs_id)
      : gtfs_id_(std::move(gtfs_id)){};
  known_addition_trip(gtfs_trip_id gtfs_id, primary_trip_id const& prim_id)
      : gtfs_id_(std::move(gtfs_id)), primary_id_(prim_id){};

  gtfs_trip_id gtfs_id_;
  primary_trip_id primary_id_;
};

struct knowledge_context {
  void sort_known_lists();

  bool is_cancel_known(transit_realtime::TripDescriptor const&) const;
  bool is_additional_known(transit_realtime::TripDescriptor const&) const;
  known_stop_skips* find_trip_stop_skips(
      transit_realtime::TripDescriptor const&) const;
  known_addition_trip& find_additional(std::string const&, unixtime);

  void remember_additional(std::string const&, unixtime, time, int);
  void update_additional(std::string const&, unixtime, time, int);
  void remember_canceled(transit_realtime::TripDescriptor const&);
  void remember_canceled(std::string const&, unixtime);
  known_stop_skips* remember_stop_skips(std::string const&, unixtime);

  std::vector<known_addition_trip> known_additional_;
  std::vector<gtfs_trip_id> known_canceled_;
  int new_known_add_cnt_{0};
  int new_known_can_cnt_{0};

  std::vector<std::unique_ptr<known_stop_skips>> known_stop_skips_;
  int new_known_skip_cnt_{0};
};

struct trip_update_context {
  trip_update_context(schedule& sched,
                      transit_realtime::TripUpdate& trip_update,
                      bool allow_addition_skip)
      : sched_(sched),
        trip_update_(trip_update),
        is_addition_skip_allowed_{allow_addition_skip} {};

  schedule& sched_;
  transit_realtime::TripUpdate& trip_update_;
  bool is_addition_skip_allowed_{true};

  trip const* trip_{nullptr};
  std::string trip_id_;
  unixtime trip_start_date_{0};

  std::vector<evt> is_events_;
  std::vector<evt> forecast_event_;
  std::vector<evt> reroute_events_;
  std::vector<evt> additional_events_;

  // is stop at vector index newly skipped in this trip update?
  std::vector<bool> is_stop_skip_new_;
  known_stop_skips* known_stop_skips_{nullptr};

  bool is_addition_{false};
  bool is_new_addition_{false};
  bool is_canceled_{false};
  bool is_new_canceled_{false};
};

}  // namespace ris::gtfsrt
}  // namespace motis
