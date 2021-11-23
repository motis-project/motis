#pragma once

#include <optional>
#include <string_view>

namespace motis::ris {

struct string_view_reader {
  explicit string_view_reader(std::string_view sv) : sv_{sv} {}

  std::optional<std::string_view> read() {
    if (read_) {
      return {};
    } else {
      read_ = true;
      return sv_;
    }
  }

  float progress() const { return read_ ? 0.0F : 1.0F; }

  static std::string_view current_file_name() { return {}; }

  std::string_view sv_;
  bool read_{false};
};

}  // namespace motis::ris
