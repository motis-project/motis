#pragma once

#include <string>

#include "conf/duration.h"
#include "rabbitmq/login.hpp"

namespace motis::ris {

struct rabbitmq_config {
  amqp::login login_;
  bool resume_stream_{true};
  unsigned update_interval_{60};  // seconds
  std::string name_{"rabbitmq"};
  conf::duration max_resume_age_{};
};

}  // namespace motis::ris
