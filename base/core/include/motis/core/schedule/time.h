#pragma once

#include <cassert>
#include <cinttypes>
#include <ctime>
#include <limits>
#include <string>
#include <tuple>

#include "cista/hash.h"

#include "motis/core/common/constexpr_abs.h"
#include "motis/core/common/unixtime.h"

namespace motis {

using day_idx_t = int16_t;
using mam_t = int16_t;
using duration_t = uint16_t;

constexpr auto const MAX_DAYS = day_idx_t{512};
constexpr auto const MINUTES_A_DAY = duration_t{1440};
constexpr auto const SECONDS_A_DAY = uint32_t{MINUTES_A_DAY * 60};
constexpr auto const INVALID_DURATION = std::numeric_limits<duration_t>::max();
constexpr auto const INVALID_MAM = std::numeric_limits<mam_t>::max();

struct time {
  constexpr time() = default;

  constexpr time(day_idx_t const day, mam_t const minute)
      : day_idx_{static_cast<decltype(day_idx_)>(
            day + static_cast<day_idx_t>(minute / MINUTES_A_DAY))},
        mam_{static_cast<decltype(mam_)>(minute % MINUTES_A_DAY)} {}

  auto cista_members() { return std::tie(day_idx_, mam_); }

  constexpr explicit time(unixtime const timestamp)
      : day_idx_{static_cast<day_idx_t>(constexpr_abs(timestamp) /
                                        MINUTES_A_DAY)},
        mam_{static_cast<mam_t>(constexpr_abs(timestamp) % MINUTES_A_DAY)} {
    if (timestamp < 0) {
      *this = -*this;
    }
  }

  constexpr inline bool valid() const {
    return day_idx_ < MAX_DAYS && mam_ < MINUTES_A_DAY;
  }

  constexpr inline int32_t ts() const {
    return day_idx_ * MINUTES_A_DAY + mam_;
  }

  time operator+(time const& o) const {
    time tmp;
    tmp.mam_ = mam_ + o.mam_;
    tmp.day_idx_ = day_idx_ + o.day_idx_ + (tmp.mam_ / MINUTES_A_DAY);
    tmp.mam_ %= MINUTES_A_DAY;
    assert(tmp.valid());
    return tmp;
  }

  time operator+(int32_t const o) const {
    auto tmp = time(ts() + o);
    assert(tmp.valid());
    return tmp;
  }

  time& operator+=(int32_t const o) {
    *this = time{ts() + o};
    return *this;
  }

  time operator-(int32_t const& o) const {
    auto tmp = time(ts() - o);
    assert(tmp.valid());
    return tmp;
  }

  time operator-() const {
    time tmp;
    if (mam_ != 0) {
      tmp.day_idx_ = -day_idx_ - static_cast<int16_t>(1);
      tmp.mam_ = MINUTES_A_DAY - mam_;
      tmp.day_idx_ -= tmp.mam_ / MINUTES_A_DAY;  // if mam_ == 0: subtract 1
    } else {
      tmp.day_idx_ = -day_idx_;
      tmp.mam_ = 0;
    }
    assert(tmp.valid());
    return tmp;
  }

  int32_t operator-(time const& o) const { return ts() - o.ts(); }

  time& operator-=(int32_t const o) {
    *this = time{ts() - o};
    return *this;
  }

  bool operator<(time const& o) const { return ts() < o.ts(); }
  bool operator>(time const& o) const { return ts() > o.ts(); }
  bool operator<=(time const& o) const { return ts() <= o.ts(); }
  bool operator>=(time const& o) const { return ts() >= o.ts(); }
  bool operator<(int32_t const& o) const { return ts() < o; }
  bool operator>(int32_t const& o) const { return ts() > o; }
  bool operator<=(int32_t const& o) const { return ts() <= o; }
  bool operator>=(int32_t const& o) const { return ts() >= o; }
  bool operator==(time const& o) const {
    return day_idx_ == o.day_idx_ && mam_ == o.mam_;
  }

  bool operator!=(time const& o) const { return !operator==(o); }

  time& operator++() {
    *this = time(ts() + 1);
    assert(this->valid());
    return *this;
  }

  time& operator--() {
    *this = time(ts() - 1);
    assert(this->valid());
    return *this;
  }

  friend std::ostream& operator<<(std::ostream& out, time const& t);

  std::string to_str() const;

  constexpr day_idx_t day() const {
    assert(day_idx_ <= MAX_DAYS);
    return day_idx_;
  }

  constexpr mam_t mam() const {
    assert(mam_ < MINUTES_A_DAY);
    return mam_;
  }

  constexpr cista::hash_t hash() const {
    return cista::hash_combine(day_idx_, mam_);
  }

private:
  day_idx_t day_idx_{MAX_DAYS};
  mam_t mam_{MINUTES_A_DAY};
};

constexpr time INVALID_TIME = time();
constexpr day_idx_t SCHEDULE_OFFSET_DAYS = 5;

// plus four days, because the maximum trip duration is 4 days
// plus one day, because the first valid motis timestamp is MINUTES_A_DAY
constexpr duration_t SCHEDULE_OFFSET_MINUTES =
    SCHEDULE_OFFSET_DAYS * MINUTES_A_DAY;

time to_motis_time(int day_index, int minutes);

time to_motis_time(int day_index, int hours, int minutes);

std::string format_time(time);

unixtime motis_to_unixtime(unixtime schedule_begin, time t);

time unix_to_motistime(unixtime schedule_begin, unixtime t);

}  // namespace motis
