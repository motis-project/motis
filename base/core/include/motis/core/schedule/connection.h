#pragma once

#include <cinttypes>
#include <tuple>

#include "motis/string.h"
#include "motis/vector.h"

#include "cista/reflection/comparable.h"

#include "motis/core/common/hash_helper.h"
#include "motis/core/schedule/attribute.h"
#include "motis/core/schedule/event_type.h"
#include "motis/core/schedule/provider.h"
#include "motis/core/schedule/time.h"
#include "motis/core/schedule/trip_idx.h"

namespace motis {

constexpr auto kMaxValidTrainNr = 99999;

using service_class_t = uint8_t;

enum class service_class : service_class_t {
  AIR = 0,
  ICE = 1,
  IC = 2,
  COACH = 3,
  N = 4,
  RE = 5,
  RB = 6,
  S = 7,
  U = 8,
  STR = 9,
  BUS = 10,
  SHIP = 11,
  OTHER = 12,
  NUM_CLASSES
};

inline service_class& operator++(service_class& c) {
  return c = static_cast<service_class>(static_cast<service_class_t>(c) + 1);
}

struct connection_info {
  CISTA_COMPARABLE();

  mcd::vector<ptr<attribute const>> attributes_;
  mcd::string line_identifier_;
  ptr<mcd::string const> dir_{nullptr};
  ptr<provider const> provider_{nullptr};
  uint32_t family_{0U};
  uint32_t train_nr_{0U};
  uint32_t original_train_nr_{0U};
  ptr<connection_info const> merged_with_{nullptr};
};

struct connection {
  ptr<connection_info const> con_info_{nullptr};
  uint16_t price_{0U};
  uint16_t d_track_{0U}, a_track_{0U};
  service_class clasz_{service_class::AIR};  // service_class 0
};

struct light_connection {
  light_connection()
      : full_con_{nullptr},
        d_time_{INVALID_TIME},
        a_time_{INVALID_TIME},
        trips_{0U},
        valid_{0U} {}

  explicit light_connection(time d_time) : d_time_{d_time} {}  // NOLINT

  light_connection(time const d_time, time const a_time,
                   connection const* full_con = nullptr,
                   merged_trips_idx const trips = 0)
      : full_con_{full_con},
        d_time_{d_time},
        a_time_{a_time},
        trips_{trips},
        valid_{1U} {}

  time event_time(event_type const t) const {
    return t == event_type::DEP ? d_time_ : a_time_;
  }

  unsigned travel_time() const { return a_time_ - d_time_; }

  inline bool operator<(light_connection const& o) const {
    return d_time_ < o.d_time_;
  }

  ptr<connection const> full_con_;
  time d_time_, a_time_;
  uint32_t trips_ : 31;
  uint32_t valid_ : 1;

#ifdef MOTIS_CAPACITY_IN_SCHEDULE
  uint16_t capacity_{};
  uint16_t passengers_{};
#endif
};

// Index of a light_connection in a route edge.
using lcon_idx_t = uint32_t;

}  // namespace motis
