#pragma once

#include <tuple>

#include "motis/core/schedule/schedule.h"

#include "motis/tripbased/data.h"

namespace motis::tripbased {

struct destination_arrival {
  destination_arrival() = default;
  destination_arrival(line_id line, stop_idx_t stop_index,
                      tb_footpath const& fp)
      : line_(line), stop_index_(stop_index), footpath_(fp) {}

  line_id line_{};
  stop_idx_t stop_index_{};
  tb_footpath footpath_{};
};

struct tb_journey {
  tb_journey() = default;
  tb_journey(search_dir dir, time start_time, time arrival_time,
             unsigned transfers, unsigned transports,
             station_id destination_station,
             destination_arrival const* dest_arrival,
             std::size_t final_queue_entry)
      : dir_(dir),
        start_time_(start_time),
        arrival_time_(arrival_time),
        duration_(arrival_time > start_time ? arrival_time - start_time
                                            : start_time - arrival_time),
        transfers_(transfers),
        transports_(transports),
        destination_station_(destination_station),
        destination_arrival_(dest_arrival),
        final_queue_entry_(final_queue_entry) {}

  bool dominates(tb_journey const& other) const {
    return (duration_ < other.duration_ && transports_ <= other.transports_) ||
           (duration_ <= other.duration_ && transports_ < other.transports_);
  }

  bool is_reconstructed() const { return !edges_.empty(); }

  station_id departure_station() const {
    return dir_ == search_dir::FWD ? start_station_ : destination_station_;
  }

  station_id arrival_station() const {
    return dir_ == search_dir::FWD ? destination_station_ : start_station_;
  }

  time departure_time() const {
    return dir_ == search_dir::FWD ? start_time_ : arrival_time_;
  }

  time arrival_time() const {
    return dir_ == search_dir::FWD ? arrival_time_ : start_time_;
  }

  time actual_departure_time() const {
    assert(!edges_.empty());
    return edges_.front().departure_time_;
  }

  time actual_arrival_time() const {
    assert(!edges_.empty());
    return edges_.back().arrival_time_;
  }

  unsigned duration() const { return arrival_time() - departure_time(); }

  unsigned actual_duration() const {
    return actual_arrival_time() - actual_departure_time();
  }

  struct tb_edge {
    tb_edge() = default;
    tb_edge(trip_id trip, stop_idx_t from_stop_index, stop_idx_t to_stop_index,
            time departure_time, time arrival_time)
        : trip_(trip),
          from_stop_index_(from_stop_index),
          to_stop_index_(to_stop_index),
          departure_time_(departure_time),
          arrival_time_(arrival_time) {}
    explicit tb_edge(tb_footpath const& footpath, time departure_time,
                     time arrival_time, int mumo_id = -1,
                     unsigned mumo_price = 0, unsigned mumo_accessibility = 0)
        : departure_time_(departure_time),
          arrival_time_(arrival_time),
          footpath_(footpath),
          mumo_id_(mumo_id),
          mumo_price_(mumo_price),
          mumo_accessibility_(mumo_accessibility) {
      assert(arrival_time_ - departure_time_ == footpath_.duration_);
    }

    bool is_connection() const { return to_stop_index_ > from_stop_index_; }
    bool is_walk() const { return !is_connection(); }

    trip_id trip_{};
    stop_idx_t from_stop_index_{};
    stop_idx_t to_stop_index_{};
    time departure_time_{INVALID_TIME};
    time arrival_time_{INVALID_TIME};
    tb_footpath footpath_{};
    int mumo_id_{-1};
    unsigned mumo_price_{0};
    unsigned mumo_accessibility_{0};
  };

  search_dir dir_{search_dir::FWD};
  time start_time_{INVALID_TIME};
  time arrival_time_{INVALID_TIME};
  unsigned duration_{};
  unsigned transfers_{};
  unsigned accessibility_{};
  unsigned transports_{};
  station_id start_station_{};
  station_id destination_station_{};
  destination_arrival const* destination_arrival_{};
  std::size_t final_queue_entry_{};
  std::vector<tb_edge> edges_;
};

}  // namespace motis::tripbased
