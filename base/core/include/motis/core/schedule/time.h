#pragma once

#include <cassert>
#include <cinttypes>
#include <ctime>
#include <limits>
#include <string>

#include "motis/core/common/constexpr_abs.h"
#include "motis/core/common/unixtime.h"

namespace motis {

using day_idx_t = int16_t;
using duration = uint16_t;

constexpr auto const MAX_DAYS = day_idx_t{512};
constexpr auto const MINUTES_A_DAY = duration{1440};
constexpr auto const SECONDS_A_DAY = uint32_t{MINUTES_A_DAY * 60};
constexpr auto const INVALID_DURATION = std::numeric_limits<duration>::max();

struct time {
  constexpr time() = default;

  constexpr time(day_idx_t const day, int16_t const minute)
      : day_{static_cast<decltype(day_)>(
            day + static_cast<day_idx_t>(minute / MINUTES_A_DAY))},
        min_{static_cast<decltype(min_)>(minute % MINUTES_A_DAY)} {}

  constexpr explicit time(int64_t const timestamp)
      : day_{static_cast<decltype(day_)>(constexpr_abs(timestamp) /
                                         MINUTES_A_DAY)},
        min_{static_cast<decltype(min_)>(constexpr_abs(timestamp) %
                                         MINUTES_A_DAY)} {
    if (timestamp < 0) {
      *this = -*this;
    }
  }

  constexpr inline bool valid() const {
    return day_ < MAX_DAYS && min_ < MINUTES_A_DAY;
  }

  constexpr inline int32_t ts() const { return day_ * MINUTES_A_DAY + min_; }

  time operator+(time const& o) const {
    time tmp;
    tmp.min_ = min_ + o.min_;
    tmp.day_ = day_ + o.day_ + (tmp.min_ / MINUTES_A_DAY);
    tmp.min_ %= MINUTES_A_DAY;
    assert(tmp.valid());
    return tmp;
  }

  time operator+(int32_t const o) const {
    auto tmp = time(ts() + o);
    assert(tmp.valid());
    return tmp;
  }

  time operator-(time const& o) const { return *this - o.ts(); }

  time operator-(int32_t const& o) const {
    auto tmp = time(ts() - o);
    assert(tmp.valid());
    return tmp;
  }

  time operator-() const {
    time tmp;
    if (min_ != 0) {
      tmp.day_ = -day_ - static_cast<int16_t>(1);
      tmp.min_ = MINUTES_A_DAY - min_;
      tmp.day_ -= tmp.min_ / MINUTES_A_DAY;  // if min_ == 0: subtract 1
    } else {
      tmp.day_ = -day_;
      tmp.min_ = 0;
    }
    assert(tmp.valid());
    return tmp;
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
    return day_ == o.day_ && min_ == o.min_;
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

  friend bool operator==(time t, int i) { return i == t.ts(); }

  friend std::ostream& operator<<(std::ostream& out, time const& t);

  std::string to_str() const;

  day_idx_t day() const {
    assert(day_ <= MAX_DAYS);
    return day_;
  }

  uint16_t mam() const {
    assert(min_ < MINUTES_A_DAY);
    return min_;
  }

private:
  day_idx_t day_{MAX_DAYS};
  uint16_t min_{MINUTES_A_DAY};
};

constexpr time INVALID_TIME = time();
constexpr uint32_t SCHEDULE_OFFSET_DAYS = 5;

// plus four days, because the maximum trip duration is 4 days
// plus one day, because the first valid motis timestamp is MINUTES_A_DAY
constexpr uint32_t SCHEDULE_OFFSET_MINUTES =
    SCHEDULE_OFFSET_DAYS * MINUTES_A_DAY;

time to_motis_time(int minutes) {
  return time(SCHEDULE_OFFSET_MINUTES + minutes);
}

time to_motis_time(int day_index, int minutes);

time to_motis_time(int day_index, int hours, int minutes);

std::string format_time(time);

inline unixtime motis_to_unixtime(unixtime schedule_begin, time t) {
  return schedule_begin + t.ts() * 60;
}

inline time unix_to_motistime(unixtime const schedule_begin, unixtime const t) {
  if (t < schedule_begin) {
    return INVALID_TIME;
  }
  auto motistime = time((t - schedule_begin) / 60);
  if (!motistime.valid()) {
    return INVALID_TIME;
  }
  return motistime;
}

}  // namespace motis
