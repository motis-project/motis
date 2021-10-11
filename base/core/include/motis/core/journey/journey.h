#pragma once

#include <ctime>
#include <string>
#include <vector>

#include "motis/core/schedule/attribute.h"
#include "motis/core/schedule/free_text.h"
#include "motis/core/schedule/time.h"
#include "motis/core/schedule/timestamp_reason.h"
#include "motis/core/journey/extern_trip.h"

namespace motis {

struct journey {
  enum class connection_status : uint8_t {
    OK,
    INTERCHANGE_INVALID,
    TRAIN_HAS_BEEN_CANCELED,
    INVALID
  };

  enum class problem_type : uint8_t {
    NO_PROBLEM,
    INTERCHANGE_TIME_VIOLATED,
    CANCELED_TRAIN
  };

  struct transport {
    unsigned from_{0}, to_{0};
    bool is_walk_{false};
    std::string name_;
    std::string category_name_;
    unsigned category_id_{0};
    unsigned clasz_{0};
    unsigned train_nr_{0};
    std::string line_identifier_;
    motis::time duration_{0};
    int mumo_id_{0};
    std::string direction_;
    std::string provider_;
    unsigned mumo_price_{0};
    unsigned mumo_accessibility_{0};
    std::string mumo_type_;
  };

  struct trip {
    unsigned from_{0}, to_{0};
    extern_trip extern_trip_;
    std::string debug_;
  };

  struct stop {
    bool exit_{false}, enter_{false};
    std::string name_;
    std::string eva_no_;
    double lat_{0}, lng_{0};

    struct event_info {
      bool valid_{false};
      unixtime timestamp_{0};
      unixtime schedule_timestamp_{0};
      timestamp_reason timestamp_reason_{timestamp_reason::SCHEDULE};
      std::string track_;
      std::string schedule_track_;
    } arrival_, departure_;
  };

  struct ranged_attribute {
    unsigned from_{0}, to_{0};
    attribute attr_;
  };

  struct ranged_free_text {
    unsigned from_{0}, to_{0};
    free_text text_;
  };

  struct problem {
    problem_type type_{problem_type::NO_PROBLEM};
    unsigned from_{0}, to_{0};
  };

  bool ok() const { return problems_.empty(); }

  unsigned duration_{0}, transfers_{0}, price_{0}, accessibility_{0};
  std::vector<stop> stops_;
  std::vector<transport> transports_;
  std::vector<trip> trips_;
  std::vector<ranged_attribute> attributes_;

  connection_status status_{connection_status::OK};
  std::vector<ranged_free_text> free_texts_;
  std::vector<problem> problems_;

  unsigned night_penalty_{0}, db_costs_{0};
};

}  // namespace motis
