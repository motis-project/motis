#pragma once

#include <system_error>

namespace motis::tiles {

namespace error {
enum error_code_t {
  ok = 0,
  database_not_available = 1,
  invalid_request = 2,
};
}  // namespace error

class error_category_impl : public std::error_category {
public:
  const char* name() const noexcept override { return "motis::tiles"; }

  std::string message(int ev) const noexcept override {
    switch (ev) {
      case error::ok: return "tiles: no error";
      case error::database_not_available:
        return "tiles: database not available";
      case error::invalid_request: return "tiles: invalid request";
      default: return "tiles: unkown error";
    }
  }
};

inline const std::error_category& error_category() {
  static error_category_impl instance;
  return instance;
}

namespace error {
inline std::error_code make_error_code(error_code_t e) noexcept {
  return std::error_code(static_cast<int>(e), error_category());
}
}  // namespace error

}  // namespace motis::tiles

namespace std {

template <>
struct is_error_code_enum<motis::tiles::error::error_code_t>
    : public std::true_type {};

}  // namespace std
