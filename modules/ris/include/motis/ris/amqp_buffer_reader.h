#pragma once

#include <cstddef>
#include <optional>
#include <string_view>
#include <vector>

#include "rabbitmq/amqp.hpp"

namespace motis::ris {

struct amqp_buffer_reader {
  explicit amqp_buffer_reader(std::vector<amqp::msg> const& msgs)
      : msgs_{msgs} {}

  std::optional<std::string_view> read() {
    if (next_index_ >= msgs_.size()) {
      return {};
    }
    auto const& msg = msgs_[next_index_];
    ++next_index_;
    return {{msg.content_.c_str(), msg.content_.size()}};
  }

  float progress() const {
    auto const count = msgs_.size();
    return count > 0 ? next_index_ / count : 1;
  }

  static std::string_view current_file_name() { return {}; }

  std::vector<amqp::msg> const& msgs_;
  std::size_t next_index_{0};
};

}  // namespace motis::ris
