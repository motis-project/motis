#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "cista/reflection/comparable.h"

#include "motis/hash_map.h"
#include "motis/string.h"
#include "motis/vector.h"

#include "motis/core/schedule/schedule.h"
#include "motis/core/schedule/time.h"

namespace motis::paxmon::eval::forecast {

enum class check_type : std::uint8_t {
  NOT_CHECKED,
  TICKED_CHECKED,
  CHECKIN,
  BOTH
};

enum class leg_status : std::uint8_t {
  NOT_CHECKED_COVERED,
  CHECKED_PLANNED,
  CHECKED_DEVIATION_EXACT_MATCH,
  CHECKED_DEVIATION_EQUIVALENT_MATCH,
  CHECKED_DEVIATION_NO_MATCH,
  NOT_CHECKED_NOT_COVERED
};

enum class travel_direction : std::uint8_t { UNKNOWN, OUTWARD, RETURN };

struct pax_check_entry {
  [[nodiscard]] inline bool has_leg_info() const {
    return leg_start_station_ != 0 && leg_destination_station_ != 0 &&
           leg_start_time_ != INVALID_TIME &&
           leg_destination_time_ != INVALID_TIME;
  }

  [[nodiscard]] inline bool has_checkin_info() const {
    return checkin_start_station_ != 0 && checkin_destination_station_ != 0;
  }

  [[nodiscard]] inline bool has_check_time() const {
    return check_min_time_ != INVALID_TIME && check_max_time_ != INVALID_TIME;
  }

  [[nodiscard]] inline bool has_schedule_train_start_time() const {
    return schedule_train_start_time_ != INVALID_TIME;
  }

  [[nodiscard]] inline bool all_checks_between(time const min,
                                               time const max) const {
    return has_check_time() && check_min_time_ >= min && check_max_time_ <= max;
  }

  [[nodiscard]] inline bool maybe_checked_between(time const min,
                                                  time const max) const {
    return has_check_time() && check_min_time_ <= max && check_max_time_ >= min;
  }

  [[nodiscard]] inline bool definitely_checked_between(time const min,
                                                       time const max) const {
    return has_check_time() &&
           ((check_min_time_ >= min && check_min_time_ <= max) ||
            (check_max_time_ >= min && check_max_time_ <= max));
  }

  [[nodiscard]] inline bool leg_between(time const min, time const max) const {
    return has_leg_info() && leg_start_time_ >= min &&
           leg_destination_time_ <= max;
  }

  [[nodiscard]] inline bool in_leg(time const dep, time const arr) const {
    return has_leg_info() && leg_start_time_ <= dep &&
           leg_destination_time_ >= arr;
  }

  std::uint64_t ref_{};

  mcd::string order_id_;
  mcd::string trip_id_;

  check_type check_type_{};
  std::uint8_t check_count_{};
  leg_status leg_status_{};
  travel_direction direction_{};
  bool planned_train_{};
  bool checked_in_train_{};
  bool canceled_{};

  std::uint32_t leg_start_station_{};  // station idx, 0 = missing
  std::uint32_t leg_destination_station_{};  // station idx, 0 = missing

  time leg_start_time_{INVALID_TIME};
  time leg_destination_time_{INVALID_TIME};

  std::uint32_t checkin_start_station_{};  // station idx, 0 = missing
  std::uint32_t checkin_destination_station_{};  // station idx, 0 = missing

  time check_min_time_{INVALID_TIME};
  time check_max_time_{INVALID_TIME};

  time schedule_train_start_time_{INVALID_TIME};

  mcd::string category_;
  std::uint32_t train_nr_{};

  std::uint64_t planned_trip_ref_{};
};

struct train_pax_data_key {
  CISTA_COMPARABLE()

  mcd::string category_;
  std::uint32_t train_nr_{};
};

struct train_pax_check_data {
  std::vector<pax_check_entry> entries_;
};

struct pax_check_data {
  void clear() {
    entries_by_order_id_.clear();
    trains_.clear();
  }

  mcd::hash_map<train_pax_data_key, train_pax_check_data> trains_;
  mcd::hash_map<mcd::string, mcd::vector<pax_check_entry const*>>
      entries_by_order_id_;
};

void load_pax_check_data(schedule const& sched, std::string const& filename,
                         pax_check_data& data);

}  // namespace motis::paxmon::eval::forecast
