#pragma once

#include "motis/core/schedule/time.h"

#include "motis/csa/csa_timetable.h"

namespace motis::csa {

struct csa_journey {
  csa_journey() = default;
  csa_journey(search_dir dir, time start_time, time arrival_time,
              unsigned transfers, csa_station const* destination_station,
              unsigned price = 0)
      : dir_(dir),
        start_time_(start_time),
        arrival_time_(arrival_time),
        duration_((arrival_time > start_time ? arrival_time - start_time
                                             : start_time - arrival_time)
                      .ts()),
        transfers_(transfers),
        price_(price),
        destination_station_(destination_station) {}

  bool is_reconstructed() const { return !edges_.empty(); }

  csa_station const* departure_station() const {
    return dir_ == search_dir::FWD ? start_station_ : destination_station_;
  }

  csa_station const* arrival_station() const {
    return dir_ == search_dir::FWD ? destination_station_ : start_station_;
  }

  struct csa_edge {
    csa_edge() = default;
    csa_edge(light_connection const* con, csa_station const* from,
             csa_station const* to, bool enter, bool exit, time departure,
             time arrival)
        : con_(con),
          from_(from),
          to_(to),
          enter_(enter),
          exit_(exit),
          departure_(departure),
          arrival_(arrival) {}
    csa_edge(csa_station const* from, csa_station const* to, time departure,
             time arrival, int mumo_id = 0, unsigned mumo_price = 0,
             unsigned mumo_accessibility = 0)
        : from_(from),
          to_(to),
          departure_(departure),
          arrival_(arrival),
          mumo_id_(mumo_id),
          mumo_price_(mumo_price),
          mumo_accessibility_(mumo_accessibility) {}

    bool is_connection() const { return con_ != nullptr; }
    bool is_walk() const { return con_ == nullptr; }
    int duration() const { return (arrival_ - departure_).ts(); }

    light_connection const* con_{nullptr};
    csa_station const* from_{nullptr};
    csa_station const* to_{nullptr};
    bool enter_{false};
    bool exit_{false};
    time departure_{INVALID_TIME};
    time arrival_{INVALID_TIME};
    int mumo_id_{0};
    unsigned mumo_price_{0};
    unsigned mumo_accessibility_{0};
  };

  friend std::ostream& operator<<(std::ostream& out, csa_journey const& j) {
    out << "{ ";
    if (j.is_reconstructed()) {
      out << "from=" << j.departure_station()->id_
          << ", to=" << j.arrival_station()->id_ << ", ";
    }
    out << format_time(j.journey_begin()) << " - "
        << format_time(j.journey_end())
        << ", duration=" << format_time(j.duration())
        << ", transfers=" << j.transfers_ << ", q_idx=" << j.query_idx_ << "  ";
    for (auto const& e : j.edges_) {
      out << "[" << (e.is_walk() ? "W" : "C") << " "
          << format_time(e.departure_) << "-" << format_time(e.arrival_) << "]";
    }
    return out << "}";
  }

  time journey_begin() const { return edges_.front().departure_; }
  time journey_end() const { return edges_.back().arrival_; }
  time duration() const { return journey_end() - journey_begin(); }

  uint32_t query_idx_{0U};
  search_dir dir_{search_dir::FWD};
  time start_time_{INVALID_TIME};
  time arrival_time_{INVALID_TIME};
  duration_t duration_{};
  unsigned transfers_{};
  unsigned accessibility_{};
  unsigned price_{};
  csa_station const* start_station_{nullptr};
  csa_station const* destination_station_{nullptr};
  std::vector<csa_edge> edges_;
};

}  // namespace motis::csa
