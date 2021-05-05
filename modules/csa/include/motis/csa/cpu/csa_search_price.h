#pragma once

#include <algorithm>
#include <array>
#include <iostream>
#include <map>

#include "utl/erase_if.h"
#include "utl/verify.h"

#include "motis/core/common/logging.h"

#include "motis/csa/csa_journey.h"
#include "motis/csa/csa_search_shared.h"
#include "motis/csa/csa_statistics.h"
#include "motis/csa/csa_timetable.h"
#include "motis/csa/error.h"

namespace motis::csa::price {

using price_t = uint16_t;

constexpr price_t MINUTE_PRICE = 8;
constexpr price_t INVALID_PRICE = std::numeric_limits<price_t>::max();
constexpr trip_id INVALID_TRIP = std::numeric_limits<trip_id>::max();

template <typename T>
inline price_t add_price(price_t base, T additional) {
  return static_cast<price_t>(std::min(
      static_cast<uint32_t>(std::numeric_limits<price_t>::max()),
      static_cast<uint32_t>(base) + static_cast<uint32_t>(additional)));
}

struct journey_pointer {
  journey_pointer() = default;
  journey_pointer(csa_connection const* enter_con,
                  csa_connection const* exit_con, footpath const* footpath,
                  time const new_time, price_t const new_price)
      : enter_con_(enter_con),
        exit_con_(exit_con),
        footpath_(footpath),
        new_time_(new_time),
        new_price_(new_price) {}

  bool valid() const {
    return enter_con_ != nullptr && exit_con_ != nullptr &&
           footpath_ != nullptr;
  }

  csa_connection const* enter_con_{nullptr};
  csa_connection const* exit_con_{nullptr};
  footpath const* footpath_{nullptr};
  time new_time_{};
  price_t new_price_{};
};

struct station_arrival_info {
  station_arrival_info() = default;
  station_arrival_info(time arrival_time, price_t price)
      : time_(arrival_time), price_(price) {}

  time time_{0};
  price_t price_{INVALID_PRICE};

  template <search_dir Dir>
  inline bool dominates(station_arrival_info const& other) const {
    return (Dir == search_dir::FWD ? time_ <= other.time_
                                   : time_ >= other.time_) &&
           dominates_price<Dir>(other);
  }

  template <search_dir Dir>
  inline bool dominates_price(station_arrival_info const& other) const {
    uint32_t const min_wage_diff = time_ > other.time_
                                       ? (time_ - other.time_) * MINUTE_PRICE
                                       : (other.time_ - time_) * MINUTE_PRICE;
    auto const fwd = Dir == search_dir::FWD;

    uint32_t const this_price =
        (time_ > other.time_) == fwd ? price_ : price_ + min_wage_diff;
    uint32_t const other_price = (time_ > other.time_) == fwd
                                     ? other.price_ + min_wage_diff
                                     : other.price_;

    return this_price <= other_price;
  }
};

template <search_dir Dir>
struct csa_search {
  static constexpr time INVALID = Dir == search_dir::FWD
                                      ? std::numeric_limits<time>::max()
                                      : std::numeric_limits<time>::min();

  csa_search(csa_timetable const& tt, time start_time, csa_statistics& stats)
      : tt_(tt),
        start_time_(start_time),
        stop_time_(INVALID),
        arrival_(tt.stations_.size(),
                 array_maker<std::vector<station_arrival_info>,
                             MAX_TRANSFERS + 1>::make_array({})),
        trip_reachable_(tt.trip_count_),
        stats_(stats) {}

  void add_start(csa_station const& station, time initial_duration,
                 price_t initial_price = 0) {
    auto const station_arrival = Dir == search_dir::FWD
                                     ? start_time_ + initial_duration
                                     : start_time_ - initial_duration;
    starts_.push_back(station.id_);
    start_times_[station.id_] = station_arrival;
    stats_.start_count_++;
    auto arrival_prices =
        array_maker<price_t, MAX_TRANSFERS + 1>::make_array(INVALID_PRICE);
    arrival_prices[0] = initial_price + initial_duration * MINUTE_PRICE;
    expand_footpaths(station, station_arrival, arrival_prices);
  }

  void search() {
    auto const& connections =
        Dir == search_dir::FWD ? tt_.fwd_connections_ : tt_.bwd_connections_;

    csa_connection const start_at{start_time_};
    auto const first_connection = std::lower_bound(
        begin(connections), end(connections), start_at,
        [&](csa_connection const& a, csa_connection const& b) {
          return Dir == search_dir::FWD ? a.departure_ < b.departure_
                                        : a.arrival_ > b.arrival_;
        });
    if (first_connection == end(connections)) {
      return;
    }

    auto const time_limit =
        Dir == search_dir::FWD
            ? std::min(static_cast<time>(start_time_ + MAX_TRAVEL_TIME),
                       stop_time_)
            : std::max(static_cast<time>(start_time_ - MAX_TRAVEL_TIME),
                       stop_time_);

    for (auto it = first_connection; it != end(connections); ++it) {
      auto const& con = *it;

      auto const time_limit_reached = Dir == search_dir::FWD
                                          ? con.departure_ > time_limit
                                          : con.arrival_ < time_limit;
      if (time_limit_reached) {
        break;
      }

      stats_.connections_scanned_++;

      auto const& via_trip = price_via_trip(con);

      auto trip_reachable_prices =
          array_maker<price_t, MAX_TRANSFERS + 1>::make_array(INVALID_PRICE);
      auto trip_reachable_prices_updated = false;

      auto arrival_prices =
          array_maker<price_t, MAX_TRANSFERS + 1>::make_array(INVALID_PRICE);
      auto arrival_prices_updated = false;

      for (auto transfers = 0; transfers < MAX_TRANSFERS; ++transfers) {
        auto const via_trip_price = via_trip[transfers];  // NOLINT
        auto const via_station_price = price_via_station(con, transfers);

        if (via_trip_price != INVALID_PRICE) {
          stats_.reachable_via_trip_++;
        }
        if (via_station_price != INVALID_PRICE) {
          stats_.reachable_via_station_++;
        }

        if (via_trip_price != INVALID_PRICE ||
            via_station_price != INVALID_PRICE) {
          auto const departure_price =
              std::min(via_trip_price, via_station_price);
          auto const via_station = via_station_price < via_trip_price;
          if (via_station) {
            trip_reachable_prices[transfers] = via_station_price;  // NOLINT
            trip_reachable_prices_updated = true;
          }
          auto const out_allowed = Dir == search_dir::FWD
                                       ? con.to_out_allowed_
                                       : con.from_in_allowed_;
          if (!out_allowed) {
            continue;
          }
          arrival_prices[transfers + 1] = add_price(  // NOLINT
              departure_price, con.price_ + con.get_duration() * MINUTE_PRICE);
          arrival_prices_updated = true;
        }
      }

      if (trip_reachable_prices_updated) {
        update_trip_reachable(con, trip_reachable_prices);
      }

      if (arrival_prices_updated) {
        stats_.footpaths_expanded_++;
        if (Dir == search_dir::FWD) {
          expand_footpaths(tt_.stations_[con.to_station_], con.arrival_,
                           arrival_prices);
        } else {
          expand_footpaths(tt_.stations_[con.from_station_], con.departure_,
                           arrival_prices);
        }
      }
    }
  }

  inline std::array<price_t, MAX_TRANSFERS + 1>& price_via_trip(
      csa_connection const& con) {
    auto& tr = trip_reachable_[con.trip_];
    if (tr.empty()) {
      tr.resize(
          tt_.trip_to_connections_[con.trip_].size(),
          array_maker<price_t, MAX_TRANSFERS + 1>::make_array(INVALID_PRICE));
      stats_.trip_price_init_++;
    }
    return tr[con.trip_con_idx_];
  }

  inline price_t price_via_station(csa_connection const& con, int transfers) {
    if (Dir == search_dir::FWD ? !con.from_in_allowed_ : !con.to_out_allowed_) {
      return INVALID_PRICE;
    }
    auto const arrival = Dir == search_dir::FWD
                             ? arrival_[con.from_station_][transfers]  // NOLINT
                             : arrival_[con.to_station_][transfers];  // NOLINT
    auto price = INVALID_PRICE;
    for (auto const& sai : arrival) {
      auto const reachable = Dir == search_dir::FWD
                                 ? sai.time_ <= con.departure_
                                 : sai.time_ >= con.arrival_;
      if (!reachable) {
        break;
      }
      auto const waiting_time = Dir == search_dir::FWD
                                    ? con.departure_ - sai.time_
                                    : sai.time_ - con.arrival_;
      price =
          std::min(price, add_price(sai.price_, waiting_time * MINUTE_PRICE));
    }
    return price;
  }

  inline void update_trip_reachable(
      csa_connection const& con,
      std::array<price_t, MAX_TRANSFERS + 1> const& initial_prices) {
    auto& tr = trip_reachable_[con.trip_];
    auto const& trip_cons = tt_.trip_to_connections_[con.trip_];
    assert(tr.size() == trip_cons.size());
    stats_.trip_reachable_updates_++;

    auto update = array_maker<bool, MAX_TRANSFERS + 1>::make_array(false);
    std::array<price_t, MAX_TRANSFERS + 1> price{};
    for (auto transfers = 0; transfers < MAX_TRANSFERS + 1; ++transfers) {
      auto const initial_price = initial_prices[transfers];  // NOLINT
      if (initial_price != INVALID_PRICE) {
        assert(tr[con.trip_con_idx_][transfers] > initial_price);  // NOLINT
        tr[con.trip_con_idx_][transfers] = initial_price;  // NOLINT
        price[transfers] = initial_price;  // NOLINT
        update[transfers] = true;  // NOLINT
      }
    }
    if (Dir == search_dir::FWD) {
      for (auto con_idx = con.trip_con_idx_ + 1UL; con_idx < trip_cons.size();
           ++con_idx) {
        auto const& prev_con = trip_cons[con_idx - 1];
        auto const& cur_con = trip_cons[con_idx];
        auto const price_delta =
            prev_con->price_ +
            (cur_con->departure_ - prev_con->departure_) * MINUTE_PRICE;
        for (auto transfers = 0; transfers < MAX_TRANSFERS + 1; ++transfers) {
          if (!update[transfers]) {  // NOLINT
            continue;
          }
          if (price[transfers] >= tr[con_idx][transfers]) {  // NOLINT
            return;
          }
          price[transfers] =  // NOLINT
              add_price(price[transfers], price_delta);  // NOLINT
          tr[con_idx][transfers] = price[transfers];  // NOLINT
        }
      }
    } else {
      if (con.trip_con_idx_ == 0) {
        return;
      }
      for (auto con_idx = con.trip_con_idx_ - 1; con_idx != -1; --con_idx) {
        auto const& prev_con = trip_cons[con_idx + 1];
        auto const& cur_con = trip_cons[con_idx];
        auto const price_delta =
            prev_con->price_ +
            (prev_con->arrival_ - cur_con->arrival_) * MINUTE_PRICE;
        for (auto transfers = 0; transfers < MAX_TRANSFERS + 1; ++transfers) {
          if (!update[transfers]) {  // NOLINT
            continue;
          }
          if (price[transfers] >= tr[con_idx][transfers]) {  // NOLINT
            return;
          }
          price[transfers] =  // NOLINT
              add_price(price[transfers], price_delta);  // NOLINT
          tr[con_idx][transfers] = price[transfers];  // NOLINT
        }
      }
    }
  }

  void expand_footpaths(
      csa_station const& station, time arrival_time,
      std::array<price_t, MAX_TRANSFERS + 1> const& arrival_prices) {
    if (Dir == search_dir::FWD) {
      for (auto const& fp : station.footpaths_) {
        auto const fp_arrival_time = arrival_time + fp.duration_;
        auto const fp_price = fp.duration_ * MINUTE_PRICE;
        auto& arrival = arrival_[fp.to_station_];
        for (auto transfers = 0; transfers <= MAX_TRANSFERS; ++transfers) {
          if (arrival_prices[transfers] == INVALID_PRICE) {  // NOLINT
            continue;
          }
          auto const fp_arrival_price =
              add_price(arrival_prices[transfers], fp_price);  // NOLINT
          station_arrival_info new_arrival(fp_arrival_time, fp_arrival_price);
          update_arrivals(arrival[transfers], new_arrival);  // NOLINT
        }
      }
    } else {
      for (auto const& fp : station.incoming_footpaths_) {
        auto const fp_arrival_time = arrival_time - fp.duration_;
        auto const fp_price = fp.duration_ * MINUTE_PRICE;
        auto& arrival = arrival_[fp.from_station_];
        for (auto transfers = 0; transfers <= MAX_TRANSFERS; ++transfers) {
          if (arrival_prices[transfers] == INVALID_PRICE) {  // NOLINT
            continue;
          }
          auto const fp_arrival_price =
              add_price(arrival_prices[transfers], fp_price);  // NOLINT
          station_arrival_info new_arrival(fp_arrival_time, fp_arrival_price);
          update_arrivals(arrival[transfers], new_arrival);  // NOLINT
        }
      }
    }
  }

  inline void update_arrivals(std::vector<station_arrival_info>& dest,
                              station_arrival_info const& new_arrival) {
    for (auto const& sai : dest) {
      if ((Dir == search_dir::FWD) ? sai.time_ > new_arrival.time_
                                   : sai.time_ < new_arrival.time_) {
        break;
      }
      if (sai.dominates_price<Dir>(new_arrival)) {
        stats_.new_labels_dominated_++;
        return;
      }
    }

    auto const prev_size = dest.size();
    utl::erase_if(dest, [&](station_arrival_info const& existing) {
      return new_arrival.dominates<Dir>(existing);
    });
    stats_.existing_labels_dominated_ += prev_size - dest.size();
    dest.insert(std::upper_bound(begin(dest), end(dest), new_arrival,
                                 [](auto const& a, auto const& b) {
                                   return Dir == search_dir::FWD
                                              ? a.time_ < b.time_
                                              : a.time_ > b.time_;
                                 }),
                new_arrival);
    stats_.labels_created_++;
    stats_.max_labels_per_station_ = std::max(
        stats_.max_labels_per_station_, static_cast<uint64_t>(dest.size()));
  }

  inline bool is_start(station_id station) const {
    return std::find(begin(starts_), end(starts_), station) != end(starts_);
  }

  std::vector<csa_journey> get_results(csa_station const& station,
                                       bool include_equivalent) {
    utl::verify_ex(!include_equivalent,
                   std::system_error{error::include_equivalent_not_supported});

    std::vector<csa_journey> journeys;
    auto const& station_arrival = arrival_[station.id_];

    auto const dominated = [&](duration dur, price_t price) {
      return std::any_of(begin(journeys), end(journeys),
                         [&](csa_journey const& j) {
                           return j.duration_ <= dur && j.price_ <= price;
                         });
    };

    for (auto transfers = 0; transfers <= MAX_TRANSFERS; ++transfers) {
      for (auto const& sai : station_arrival[transfers]) {  // NOLINT
        auto const dur = sai.time_ > start_time_ ? sai.time_ - start_time_
                                                 : start_time_ - sai.time_;
        if (!dominated(dur, sai.price_)) {
          journeys.emplace_back(Dir, start_time_, sai.time_, transfers,
                                &station, sai.price_);
        }
      }
    }
    return journeys;
  }

  void extract_journey(csa_journey& j) {
    if (j.is_reconstructed()) {
      return;
    }
    auto stop = j.destination_station_;
    auto transfers = j.transfers_;
    auto t = j.arrival_time_;
    auto price = static_cast<price_t>(j.price_);
    for (; transfers > 0; --transfers) {
      auto const jp = get_journey_pointer(*stop, t, transfers, price);
      if (jp.valid()) {
        if (jp.footpath_->from_station_ != jp.footpath_->to_station_) {
          if (Dir == search_dir::FWD) {
            j.edges_.emplace_back(
                &tt_.stations_[jp.footpath_->from_station_],
                &tt_.stations_[jp.footpath_->to_station_],
                jp.exit_con_->arrival_,
                jp.exit_con_->arrival_ + jp.footpath_->duration_, -1);
          } else {
            j.edges_.emplace_back(
                &tt_.stations_[jp.footpath_->from_station_],
                &tt_.stations_[jp.footpath_->to_station_],
                jp.enter_con_->departure_ - jp.footpath_->duration_,
                jp.enter_con_->departure_, -1);
          }
        }
        assert(jp.enter_con_->trip_ == jp.exit_con_->trip_);
        auto const& trip_cons = tt_.trip_to_connections_[jp.exit_con_->trip_];
        auto const add_trip_edge = [&](csa_connection const* con) {
          auto const enter = con == jp.enter_con_;
          auto const exit = con == jp.exit_con_;
          j.edges_.emplace_back(con->light_con_,
                                &tt_.stations_[con->from_station_],
                                &tt_.stations_[con->to_station_], enter, exit,
                                con->departure_, con->arrival_);
        };
        if (Dir == search_dir::FWD) {
          auto in_trip = false;
          for (int i = static_cast<int>(trip_cons.size()) - 1; i >= 0; --i) {
            auto const con = trip_cons[i];
            if (con == jp.exit_con_) {
              in_trip = true;
            }
            if (in_trip) {
              add_trip_edge(con);
            }
            if (con == jp.enter_con_) {
              break;
            }
          }
          stop = &tt_.stations_[jp.enter_con_->from_station_];
        } else {
          auto in_trip = false;
          for (auto const& con : trip_cons) {
            if (con == jp.enter_con_) {
              in_trip = true;
            }
            if (in_trip) {
              add_trip_edge(con);
            }
            if (con == jp.exit_con_) {
              break;
            }
          }
          stop = &tt_.stations_[jp.exit_con_->to_station_];
        }
        j.start_station_ = stop;
        t = jp.new_time_;
        price = jp.new_price_;
      } else {
        if (!is_start(stop->id_)) {
          if (transfers != 0) {
            LOG(motis::logging::warn)
                << "csa extract journey: adding final footpath "
                   "with transfers="
                << transfers;
          }
          add_final_footpath(j, stop, t, transfers, price);
        }
        break;
      }
    }
    if (transfers == 0 && !is_start(stop->id_)) {
      add_final_footpath(j, stop, t, transfers, price);
    }
    if (Dir == search_dir::FWD) {
      std::reverse(begin(j.edges_), end(j.edges_));
    }
    utl::verify(!j.edges_.empty(), "csa price journey reconstruction failed");
  }

  void add_final_footpath(csa_journey& j, csa_station const* stop,
                          time arrival_time, int transfers, price_t price) {
    (void)price;
    (void)transfers;
    assert(transfers == 0);
    if (Dir == search_dir::FWD) {
      for (auto const& fp : stop->incoming_footpaths_) {
        if (fp.from_station_ == fp.to_station_) {
          continue;
        }
        auto const fp_departure = arrival_time - fp.duration_;
        auto const valid_station = is_start(fp.from_station_);
        auto const valid_time = fp_departure >= start_times_[fp.from_station_];
        if (valid_station && valid_time) {
          j.edges_.emplace_back(&tt_.stations_[fp.from_station_],
                                &tt_.stations_[fp.to_station_], fp_departure,
                                arrival_time, -1);
          j.start_station_ = &tt_.stations_[fp.to_station_];
          break;
        }
      }
    } else {
      for (auto const& fp : stop->footpaths_) {
        if (fp.from_station_ == fp.to_station_) {
          continue;
        }
        auto const fp_departure = arrival_time + fp.duration_;
        auto const valid_station = is_start(fp.to_station_);
        auto const valid_time = fp_departure <= start_times_[fp.to_station_];
        if (valid_station && valid_time) {
          j.edges_.emplace_back(&tt_.stations_[fp.from_station_],
                                &tt_.stations_[fp.to_station_], arrival_time,
                                fp_departure, -1);
          j.start_station_ = &tt_.stations_[fp.from_station_];
          break;
        }
      }
    }
  }

  journey_pointer get_journey_pointer(csa_station const& station,
                                      time arrival_time, int transfers,
                                      price_t price) const {
    auto const& station_arrival = arrival_[station.id_][transfers];  // NOLINT
    if (Dir == search_dir::FWD) {
      for (auto const& arrival : station_arrival) {
        if (arrival.time_ > arrival_time) {
          continue;
        }
        auto const arrival_price =
            price - (arrival_time - arrival.time_) * MINUTE_PRICE;
        if (arrival.price_ != arrival_price) {
          continue;
        }
        for (auto const& fp : station.incoming_footpaths_) {
          auto const& fp_dep_stop = tt_.stations_[fp.from_station_];
          auto const con_arrival_time = arrival.time_ - fp.duration_;
          auto const con_arrival_price =
              arrival_price - fp.duration_ * MINUTE_PRICE;
          auto const exit_candidates = get_exit_candidates(
              fp_dep_stop, con_arrival_time, transfers, con_arrival_price);

          for (auto const& exit_con : exit_candidates) {
            auto const enter_con = get_enter_connection(
                exit_con, con_arrival_time, transfers, con_arrival_price);
            if (enter_con == nullptr) {
              continue;
            }
            assert(!trip_reachable_[enter_con->trip_].empty());
            return {enter_con, exit_con, &fp, enter_con->departure_,
                    trip_reachable_[enter_con->trip_][enter_con->trip_con_idx_]
                                   [transfers - 1]};  // NOLINT
          }
        }
      }
    } else {
      for (auto const& arrival : station_arrival) {
        if (arrival.time_ < arrival_time) {
          continue;
        }
        auto const arrival_price =
            price - (arrival.time_ - arrival_time) * MINUTE_PRICE;
        if (arrival.price_ != arrival_price) {
          continue;
        }
        for (auto const& fp : station.footpaths_) {
          auto const& fp_arr_stop = tt_.stations_[fp.to_station_];
          auto const con_arrival_time = arrival.time_ + fp.duration_;
          auto const con_arrival_price =
              arrival_price - fp.duration_ * MINUTE_PRICE;
          auto const enter_candidates = get_enter_candidates(
              fp_arr_stop, con_arrival_time, transfers, con_arrival_price);

          for (auto const& enter_con : enter_candidates) {
            auto const exit_con = get_exit_connection(
                enter_con, con_arrival_time, transfers, con_arrival_price);
            if (exit_con == nullptr) {
              continue;
            }
            assert(!trip_reachable_[exit_con->trip_].empty());
            return {enter_con, exit_con, &fp, exit_con->arrival_,
                    trip_reachable_[exit_con->trip_][exit_con->trip_con_idx_]
                                   [transfers - 1]};  // NOLINT
          }
        }
      }
    }
    return {};
  }

  std::vector<csa_connection const*> get_exit_candidates(
      csa_station const& arrival_station, time arrival_time, int transfers,
      price_t arrival_price) const {
    std::vector<csa_connection const*> candidates;

    std::copy_if(begin(arrival_station.incoming_connections_),
                 end(arrival_station.incoming_connections_),
                 std::back_inserter(candidates),
                 [&](csa_connection const* con) {
                   auto const& tr = trip_reachable_[con->trip_];
                   return con->arrival_ == arrival_time && !tr.empty() &&
                          tr[con->trip_con_idx_][transfers - 1] ==  // NOLINT
                              arrival_price - con->price_ -
                                  con->get_duration() * MINUTE_PRICE;
                 });

    return candidates;
  }

  csa_connection const* get_enter_connection(csa_connection const* exit_con,
                                             time arrival_time, int transfers,
                                             price_t arrival_price) const {
    assert(arrival_time == exit_con->arrival_);
    (void)arrival_time;
    auto const& cons = tt_.trip_to_connections_[exit_con->trip_];
    auto const& tr = trip_reachable_[exit_con->trip_];
    assert(!tr.empty());

    auto departure_price = arrival_price - exit_con->price_ -
                           exit_con->get_duration() * MINUTE_PRICE;
    auto enter_con_idx = 0;
    for (auto i = exit_con->trip_con_idx_;; --i) {
      if (tr[i][transfers - 1] == departure_price) {  // NOLINT
        enter_con_idx = i;
        if (i == 0) {
          break;
        }
        departure_price -=
            cons[i - 1]->price_ +
            (cons[i]->departure_ - cons[i - 1]->departure_) * MINUTE_PRICE;
      } else {
        break;
      }
    }
    return cons[enter_con_idx];
  }

  std::vector<csa_connection const*> get_enter_candidates(
      csa_station const& departure_station, time departure_time, int transfers,
      price_t departure_price) const {
    std::vector<csa_connection const*> candidates;

    std::copy_if(begin(departure_station.outgoing_connections_),
                 end(departure_station.outgoing_connections_),
                 std::back_inserter(candidates),
                 [&](csa_connection const* con) {
                   auto const& tr = trip_reachable_[con->trip_];
                   return con->departure_ == departure_time && !tr.empty() &&
                          tr[con->trip_con_idx_][transfers - 1] ==  // NOLINT
                              departure_price - con->price_ -
                                  con->get_duration() * MINUTE_PRICE;
                 });

    return candidates;
  }

  csa_connection const* get_exit_connection(csa_connection const* enter_con,
                                            time departure_time, int transfers,
                                            price_t departure_price) const {
    assert(departure_time == enter_con->departure_);
    (void)departure_time;
    auto const& cons = tt_.trip_to_connections_[enter_con->trip_];
    auto const& tr = trip_reachable_[enter_con->trip_];
    assert(!tr.empty());

    auto arrival_price = departure_price - enter_con->price_ -
                         enter_con->get_duration() * MINUTE_PRICE;
    auto exit_con_idx = 0;
    for (auto i = enter_con->trip_con_idx_;; ++i) {
      if (tr[i][transfers - 1] == arrival_price) {  // NOLINT
        exit_con_idx = i;
        if (i == cons.size() - 1) {
          break;
        }
        arrival_price -=
            cons[i + 1]->price_ +
            (cons[i + 1]->arrival_ - cons[i]->arrival_) * MINUTE_PRICE;
      } else {
        break;
      }
    }
    return cons[exit_con_idx];
  }

  csa_timetable const& tt_;
  time start_time_;
  time stop_time_;
  std::map<station_id, time> start_times_;
  std::vector<std::array<std::vector<station_arrival_info>, MAX_TRANSFERS + 1>>
      arrival_;
  std::vector<std::vector<std::array<price_t, MAX_TRANSFERS + 1>>>
      trip_reachable_;
  std::vector<station_id> starts_;
  csa_statistics& stats_;
};

}  // namespace motis::csa::price
