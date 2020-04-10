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

enum {
  MOTIS_ICE = 0,
  MOTIS_IC = 1,
  MOTIS_N = 2,
  MOTIS_RE = 3,
  MOTIS_RB = 4,
  MOTIS_S = 5,
  MOTIS_U = 6,
  MOTIS_STR = 7,
  MOTIS_BUS = 8,
  MOTIS_X = 9,
  NUM_CLASSES
};

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
  uint8_t clasz_{0U};
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
};

// Index of a light_connection in a route edge.
using lcon_idx_t = uint32_t;

}  // namespace motis
