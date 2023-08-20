#pragma once

#include <functional>
#include <vector>

#include "rabbitmq/amqp.hpp"

#include "motis/core/common/unixtime.h"

#include "motis/ris/rabbitmq_config.h"
#include "motis/ris/source_status.h"

namespace motis::ris::ribasis {

std::string get_queue_id(amqp::login const& login);

struct receiver {
  using msg_handler_fn =
      std::function<void(receiver&, std::vector<amqp::msg>&&)>;

  receiver(rabbitmq_config config, source_status& status,
           msg_handler_fn msg_handler);

  void stop();

  std::string const& queue_id() const;
  std::string const& name() const;

private:
  void log(std::string const& log_msg);
  void on_msg(amqp::msg const& m);

  rabbitmq_config config_;
  amqp::ssl_connection connection_;
  unixtime last_update_{};
  std::vector<amqp::msg> buffer_;
  msg_handler_fn msg_handler_;
  std::string queue_id_;

public:
  source_status& status_;
};

}  // namespace motis::ris::ribasis
