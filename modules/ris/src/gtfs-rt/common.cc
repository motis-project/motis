#include "motis/ris/gtfs-rt/common.h"

#include "motis/core/common/date_time_util.h"
#include "motis/core/common/unixtime.h"
#include "motis/core/schedule/schedule.h"
#include "motis/core/schedule/trip.h"
#include "motis/ris/gtfs-rt/parse_stop.h"
#include "motis/ris/gtfs-rt/parse_time.h"

namespace motis::ris::gtfsrt {

evt::evt(trip const& trip, stop_context const& s, event_type const type)
    : stop_idx_(s.idx_),
      seq_no_(s.seq_no_),
      stop_id_(s.station_id_),
      train_nr_(trip.id_.primary_.train_nr_),
      line_id_(trip.id_.secondary_.line_id_),
      type_(type),
      orig_sched_time_{type_ == event_type::ARR ? s.stop_arrival_
                                                : s.stop_departure_},
      new_sched_time_{orig_sched_time_} {}

known_addition_trip const& knowledge_context::find_additional(
    gtfs_trip_id const& trip_id) const {
  return const_cast<knowledge_context*>(this)  // NOLINT
      ->find_additional(trip_id);
}

known_addition_trip& knowledge_context::find_additional(
    gtfs_trip_id const& trip_id) {
  auto const lb = std::lower_bound(begin(known_additional_),
                                   end(known_additional_) - new_known_add_cnt_,
                                   known_addition_trip{trip_id});
  if (lb == end(known_additional_) || lb->gtfs_id_ != trip_id) {
    throw std::runtime_error(
        "Tried to find an unknown additional trip which should be known!");
  }
  return *lb;
}

void knowledge_context::sort_known_lists() {
  std::sort(begin(known_additional_), end(known_additional_));
  std::sort(begin(known_canceled_), end(known_canceled_));
  std::sort(begin(known_stop_skips_), end(known_stop_skips_),
            [](auto const& lhs, auto const& rhs) -> bool {
              return lhs->trip_id_ < rhs->trip_id_;
            });
  new_known_can_cnt_ = 0;
  new_known_add_cnt_ = 0;
  new_known_skip_cnt_ = 0;
}

bool knowledge_context::is_additional_known(
    transit_realtime::TripDescriptor const& d) const {
  return std::binary_search(begin(known_additional_),
                            end(known_additional_) - new_known_add_cnt_,
                            known_addition_trip(to_trip_id(d, tag_)));
}

bool knowledge_context::is_cancel_known(
    transit_realtime::TripDescriptor const& d) const {
  return std::binary_search(begin(known_canceled_),
                            end(known_canceled_) - new_known_can_cnt_,
                            to_trip_id(d, tag_));
}

known_stop_skips* knowledge_context::find_trip_stop_skips(
    transit_realtime::TripDescriptor const& d) const {
  auto const trip_id = to_trip_id(d, tag_);
  auto const lower = std::lower_bound(
      begin(known_stop_skips_), end(known_stop_skips_) - new_known_skip_cnt_,
      trip_id,
      [](std::unique_ptr<known_stop_skips> const& lhs,
         gtfs_trip_id const& rhs) { return lhs->trip_id_ < rhs; });
  if (lower == end(known_stop_skips_) || lower->get()->trip_id_ != trip_id) {
    return nullptr;
  }
  return lower->get();
}

void knowledge_context::remember_additional(gtfs_trip_id trip_id,
                                            const time start_date,
                                            const int first_station_id) {
  known_additional_.emplace_back(known_addition_trip{
      std::move(trip_id), primary_trip_id(first_station_id, 0, start_date)});
  ++new_known_add_cnt_;
}

void knowledge_context::update_additional(gtfs_trip_id const& trip_id,
                                          time const start_time,
                                          int const first_station_idx) {
  auto& additional = find_additional(trip_id);
  additional.primary_id_.station_id_ = first_station_idx;
  additional.primary_id_.time_ = start_time;
}

void knowledge_context::remember_canceled(
    transit_realtime::TripDescriptor const& d) {
  known_canceled_.emplace_back(to_trip_id(d, tag_));
  ++new_known_can_cnt_;
}

void knowledge_context::remember_canceled(gtfs_trip_id trip_id) {
  known_canceled_.emplace_back(std::move(trip_id));
  ++new_known_can_cnt_;
}

known_stop_skips* knowledge_context::remember_stop_skips(gtfs_trip_id trip) {
  ++new_known_skip_cnt_;
  return known_stop_skips_
      .emplace_back(std::make_unique<known_stop_skips>(std::move(trip)))
      .get();
}

}  // namespace motis::ris::gtfsrt
