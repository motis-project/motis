#include "motis/ris/ribasis/ribasis_receiver.h"

#include "fmt/format.h"

#include "motis/core/common/date_time_util.h"
#include "motis/core/common/logging.h"

using namespace motis::logging;

namespace motis::ris::ribasis {

std::string get_queue_id(amqp::login const& login) {
  return fmt::format("{}:{}/{}/{}", login.host_, login.port_, login.vhost_,
                     login.queue_);
}

receiver::receiver(rabbitmq_config config, source_status& status,
                   msg_handler_fn msg_handler)
    : config_{std::move(config)},
      connection_{&config_.login_,
                  [this](std::string const& log_msg) { log(log_msg); }},
      last_update_{now()},
      msg_handler_{std::move(msg_handler)},
      queue_id_{get_queue_id(config_.login_)},
      status_{status} {
  status_.enabled_ = true;
  status_.update_interval_ = config_.update_interval_;
  connection_.run([this](amqp::msg const& m) { on_msg(m); });
}

void receiver::log(std::string const& log_msg) {
  LOG(info) << config_.name_ << ": " << log_msg;
}

void receiver::on_msg(amqp::msg const& m) {
  buffer_.emplace_back(m);

  auto const n = now();
  if (n - last_update_ >= config_.update_interval_) {
    last_update_ = n;

    auto msgs = buffer_;
    buffer_.clear();
    if (msg_handler_) {
      msg_handler_(*this, std::move(msgs));
    }
  }
}

void receiver::stop() { connection_.stop(); }

std::string const& receiver::queue_id() const { return queue_id_; }

std::string const& receiver::name() const { return config_.name_; }

}  // namespace motis::ris::ribasis
