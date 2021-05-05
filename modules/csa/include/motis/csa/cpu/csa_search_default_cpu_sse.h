#pragma once

static_assert(__AVX__, "AVX not enabled!");

#include <emmintrin.h>
#include <smmintrin.h>
#include <tmmintrin.h>
#include <algorithm>
#include <iostream>
#include <limits>
#include <map>

#include "boost/align/aligned_allocator.hpp"

#include "utl/erase_if.h"
#include "utl/verify.h"

#include "motis/core/common/logging.h"

#include "motis/csa/csa_journey.h"
#include "motis/csa/csa_reconstruction.h"
#include "motis/csa/csa_search_shared.h"
#include "motis/csa/csa_statistics.h"
#include "motis/csa/csa_timetable.h"
#include "motis/csa/error.h"

namespace motis::csa::cpu::sse {

template <typename T>
using aligned_vector =
    std::vector<T, boost::alignment::aligned_allocator<T, 16>>;

// 128 bit
static_assert(MAX_TRANSFERS == 7);
static_assert(sizeof(time) == 2);

template <search_dir Dir>
struct csa_search {
  static constexpr time INVALID = Dir == search_dir::FWD
                                      ? std::numeric_limits<time>::max()
                                      : std::numeric_limits<time>::min();

  csa_search(csa_timetable const& tt, time start_time, csa_statistics& stats)
      : tt_(tt),
        start_time_(start_time),
        stop_time_(INVALID),
        arrival_time_(
            tt.stations_.size(),
            array_maker<time, MAX_TRANSFERS + 1>::make_array(INVALID)),
        trip_reachable_(tt.trip_count_),
        stats_(stats) {}

  void add_start(csa_station const& station, time initial_duration) {
    auto const station_arrival = Dir == search_dir::FWD
                                     ? start_time_ + initial_duration
                                     : start_time_ - initial_duration;
    start_times_[station.id_] = station_arrival;
    arrival_time_[station.id_][0] = station_arrival;
    stats_.start_count_++;
    expand_footpaths(station, station_arrival,
                     _mm_setr_epi16(static_cast<int16_t>(
                                        std::numeric_limits<uint16_t>::max()),
                                    0, 0, 0, 0, 0, 0, 0));
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

    auto const m_signed_offset = _mm_set1_epi16(static_cast<int16_t>(0x8000));

    for (auto it = first_connection; it != end(connections); ++it) {
      auto const& con = *it;

      auto& trip_reachable = trip_reachable_[con.trip_];
      auto& from_arrival_time = arrival_time_[con.from_station_];
      auto& to_arrival_time = arrival_time_[con.to_station_];

      auto const time_limit_reached = Dir == search_dir::FWD
                                          ? con.departure_ > time_limit
                                          : con.arrival_ < time_limit;
      if (time_limit_reached) {
        break;
      }

      stats_.connections_scanned_++;

      auto const m_via_trip =
          _mm_load_si128(reinterpret_cast<__m128i*>(trip_reachable.data()));
      auto m_reachable = m_via_trip;

      if (Dir == search_dir::FWD && con.from_in_allowed_) {
        // from_arrival_time <= con.departure
        auto const m_from_arrival_time = _mm_load_si128(
            reinterpret_cast<__m128i*>(from_arrival_time.data()));
        auto const m_con_departure_time = _mm_set1_epi16(con.departure_);
        auto const m_via_station = _mm_cmpeq_epi16(
            _mm_subs_epu16(m_from_arrival_time, m_con_departure_time),
            _mm_setzero_si128());
        m_reachable = _mm_or_si128(m_via_trip, m_via_station);
      } else if (Dir == search_dir::BWD && con.to_out_allowed_) {
        // to_arrival_time >= con.arrival == con.arrival <= to_arrival_time
        auto const m_to_arrival_time =
            _mm_load_si128(reinterpret_cast<__m128i*>(to_arrival_time.data()));
        auto const m_con_arrival_time = _mm_set1_epi16(con.arrival_);
        auto const m_via_station = _mm_cmpeq_epi16(
            _mm_subs_epu16(m_con_arrival_time, m_to_arrival_time),
            _mm_setzero_si128());
        m_reachable = _mm_or_si128(m_via_trip, m_via_station);
      }
      _mm_store_si128(reinterpret_cast<__m128i*>(trip_reachable.data()),
                      m_reachable);

      if ((Dir == search_dir::FWD && !con.to_out_allowed_) ||
          (Dir == search_dir::BWD && !con.from_in_allowed_)) {
        continue;
      }

      __m128i m_improved_arrival;
      if (Dir == search_dir::FWD) {
        // update = reachable && con.arrival < to_arrival[transfers + 1]
        auto const m_to_arrival_time =
            _mm_load_si128(reinterpret_cast<__m128i*>(to_arrival_time.data()));
        auto const m_to_arrival_time_shifted =
            _mm_srli_si128(m_to_arrival_time, 2);  // NOLINT
        auto const m_con_arrival_time_s = _mm_set1_epi16(
            static_cast<int16_t>(static_cast<int>(con.arrival_) - 0x8000));
        m_improved_arrival = _mm_cmpgt_epi16(
            _mm_sub_epi16(m_to_arrival_time_shifted, m_signed_offset),
            m_con_arrival_time_s);
      } else {
        // update = reachable && con.departure > from_arrival[transfers + 1]
        auto const m_from_arrival_time = _mm_load_si128(
            reinterpret_cast<__m128i*>(from_arrival_time.data()));
        auto const m_from_arrival_time_shifted =
            _mm_srli_si128(m_from_arrival_time, 2);  // NOLINT
        auto const m_con_departure_time_s = _mm_set1_epi16(
            static_cast<int16_t>(static_cast<int>(con.departure_) - 0x8000));
        m_improved_arrival = _mm_cmpgt_epi16(
            m_con_departure_time_s,
            _mm_sub_epi16(m_from_arrival_time_shifted, m_signed_offset));
      }
      auto const m_update = _mm_slli_si128(  // NOLINT
          _mm_and_si128(m_reachable, m_improved_arrival), 2);
      auto const any_updates = _mm_testz_si128(m_update, m_update) == 0;

      if (any_updates) {
        if (Dir == search_dir::FWD) {
          expand_footpaths(tt_.stations_[con.to_station_], con.arrival_,
                           m_update);
        } else {
          expand_footpaths(tt_.stations_[con.from_station_], con.departure_,
                           m_update);
        }
      }
    }
  }

  void expand_footpaths(csa_station const& station, time const station_arrival,
                        __m128i const& m_update) {
    stats_.footpaths_expanded_++;

    if (Dir == search_dir::FWD) {
      auto const all_zeroes = _mm_setzero_si128();
      auto const all_ones = _mm_cmpeq_epi16(all_zeroes, all_zeroes);
      for (auto const& fp : station.footpaths_) {
        // fp_arrival = (~m_update & ~0) | (fp.arrival & update)
        // arrival = min(arrival, fp_arrival)
        auto& arrival = arrival_time_[fp.to_station_];
        auto const no_update = _mm_andnot_si128(m_update, all_ones);
        auto const fp_arrival = _mm_or_si128(
            _mm_and_si128(_mm_set1_epi16(station_arrival + fp.duration_),
                          m_update),
            no_update);
        auto const old_arrival =
            _mm_load_si128(reinterpret_cast<__m128i*>(arrival.data()));
        auto const new_arrival = _mm_min_epu16(old_arrival, fp_arrival);
        _mm_store_si128(reinterpret_cast<__m128i*>(arrival.data()),
                        new_arrival);
      }
    } else {
      for (auto const& fp : station.incoming_footpaths_) {
        // arrival = max(arrival, (fp.arrival & update))
        auto& arrival = arrival_time_[fp.from_station_];
        auto const fp_arrival = _mm_and_si128(
            _mm_set1_epi16(station_arrival - fp.duration_), m_update);
        auto const old_arrival =
            _mm_load_si128(reinterpret_cast<__m128i*>(arrival.data()));
        auto const new_arrival = _mm_max_epu16(old_arrival, fp_arrival);
        _mm_store_si128(reinterpret_cast<__m128i*>(arrival.data()),
                        new_arrival);
      }
    }
  }

  std::vector<csa_journey> get_results(csa_station const& station,
                                       bool include_equivalent) {
    utl::verify_ex(!include_equivalent,
                   std::system_error{error::include_equivalent_not_supported});

    std::vector<csa_journey> journeys;
    auto const& station_arrival = arrival_time_[station.id_];
    for (auto i = 0; i <= MAX_TRANSFERS; ++i) {
      auto const arrival_time = station_arrival[i];  // NOLINT
      if (arrival_time != INVALID) {
        csa_reconstruction<Dir, decltype(arrival_time_),
                           decltype(trip_reachable_)>{
            tt_, start_times_, arrival_time_, trip_reachable_}
            .extract_journey(journeys.emplace_back(Dir, start_time_,
                                                   arrival_time, i, &station));
      }
    }
    return journeys;
  }

  csa_timetable const& tt_;
  time start_time_;
  time stop_time_;
  std::map<station_id, time> start_times_;
  aligned_vector<std::array<time, MAX_TRANSFERS + 1>> arrival_time_;
  aligned_vector<std::array<uint16_t, MAX_TRANSFERS + 1>> trip_reachable_;
  csa_statistics& stats_;
};

}  // namespace motis::csa::cpu::sse
