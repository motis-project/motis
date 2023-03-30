#pragma once

#include <string>

#include "rabbitmq/login.hpp"

namespace motis::ris {

struct rabbitmq_config {
  amqp::login login_;
  bool resume_stream_{true};
  unsigned update_interval_{60};  // seconds
  std::string name_{"rabbitmq"};
};

}  // namespace motis::ris
