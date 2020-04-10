#include "motis/ris/gtfs-rt/common.h"

#include "motis/core/common/date_time_util.h"
#include "motis/core/schedule/schedule.h"
#include "motis/core/schedule/trip.h"
#include "motis/ris/gtfs-rt/parse_stop.h"
#include "motis/ris/gtfs-rt/parse_time.h"

using trip_ids_it =
    std::vector<std::pair<motis::gtfs_trip_id, motis::trip*>>::iterator;
using trip_id_pair = std::pair<motis::gtfs_trip_id, motis::trip*>;

namespace motis::ris::gtfsrt {

evt::evt(trip const& trip, stop_context const& s, event_type const type)
    : stop_idx_(s.idx_),
      seq_no_(s.seq_no_),
      stop_id_(s.station_id_),
      train_nr_(trip.id_.primary_.train_nr_),
      line_id_(trip.id_.secondary_.line_id_),
      type_(type) {
  orig_sched_time_ =
      type_ == event_type::ARR ? s.stop_arrival_ : s.stop_departure_;
  new_sched_time_ = orig_sched_time_;
}

known_addition_trip& knowledge_context::find_additional(
    std::string const& trip_id, std::time_t const start_date) {
  auto const id = gtfs_trip_id{trip_id, start_date};
  auto lb = std::lower_bound(begin(known_additional_),
                             end(known_additional_) - new_known_add_cnt_,
                             known_addition_trip{id});
  if (lb == end(known_additional_) || lb->gtfs_id_.trip_id_ != trip_id ||
      lb->gtfs_id_.start_date_ != start_date) {
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
              return lhs->trip_id < rhs->trip_id ||
                     (lhs->trip_id == rhs->trip_id &&
                      lhs->trip_date < rhs->trip_date);
            });
  new_known_can_cnt_ = 0;
  new_known_add_cnt_ = 0;
  new_known_skip_cnt_ = 0;
}

bool knowledge_context::is_additional_known(
    transit_realtime::TripDescriptor const& descriptor) const {
  auto& trip_id = descriptor.trip_id();
  auto start_date = to_unix_time(parse_date(descriptor.start_date()));
  return std::binary_search(begin(known_additional_),
                            end(known_additional_) - new_known_add_cnt_,
                            known_addition_trip({trip_id, start_date}));
}

bool knowledge_context::is_cancel_known(
    transit_realtime::TripDescriptor const& descriptor) const {
  auto& trip_id = descriptor.trip_id();
  auto start_date = to_unix_time(parse_date(descriptor.start_date()));
  return std::binary_search(begin(known_canceled_),
                            end(known_canceled_) - new_known_can_cnt_,
                            gtfs_trip_id{trip_id, start_date});
}

known_stop_skips* knowledge_context::find_trip_stop_skips(
    transit_realtime::TripDescriptor const& descriptor) const {
  auto& trip_id = descriptor.trip_id();
  auto start_date = to_unix_time(parse_date(descriptor.start_date()));

  auto lower = std::lower_bound(
      begin(known_stop_skips_), end(known_stop_skips_) - new_known_skip_cnt_,
      std::make_pair(trip_id, start_date),
      [](auto const& lhs, std::pair<std::string, int> const& rhs) {
        return lhs->trip_id < rhs.first ||
               (lhs->trip_id == rhs.first && lhs->trip_date < rhs.second);
      });
  if (lower == end(known_stop_skips_) || lower->get()->trip_id != trip_id ||
      lower->get()->trip_date != start_date) {
    return nullptr;
  }

  return lower->get();
}

void knowledge_context::remember_additional(std::string const& trip_id,
                                            const std::time_t start_date_unix,
                                            const time start_date,
                                            const int first_station_id) {
  known_additional_.emplace_back(
      known_addition_trip{{trip_id, start_date_unix},
                          primary_trip_id(first_station_id, 0, start_date)});
  ++new_known_add_cnt_;
}

void knowledge_context::update_additional(std::string const& trip_id,
                                          const std::time_t start_date_unix,
                                          const time start_time,
                                          const int first_station_idx) {
  auto& additional = find_additional(trip_id, start_date_unix);
  additional.primary_id_.station_id_ = first_station_idx;
  additional.primary_id_.time_ = start_time;
}

void knowledge_context::remember_canceled(
    transit_realtime::TripDescriptor const& descriptor) {
  known_canceled_.emplace_back(gtfs_trip_id{
      descriptor.trip_id(), to_unix_time(parse_date(descriptor.start_date()))});
  ++new_known_can_cnt_;
}

void knowledge_context::remember_canceled(std::string const& trip_id,
                                          std::time_t const start_date) {
  known_canceled_.emplace_back(gtfs_trip_id{trip_id, start_date});
  ++new_known_can_cnt_;
}

known_stop_skips* knowledge_context::remember_stop_skips(
    std::string const& trip_id, std::time_t const date) {
  ++new_known_skip_cnt_;
  return known_stop_skips_
      .emplace_back(std::make_unique<known_stop_skips>(trip_id, date))
      .get();
}

}  // namespace motis::ris::gtfsrt