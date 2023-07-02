#pragma once

#include <system_error>
#include <type_traits>

namespace motis::paxforecast {

namespace error {
enum error_code_t {
  ok = 0,
  unsupported_measure = 1,
  universe_not_found = 2,
  invalid_rt_update_message = 3
};
}  // namespace error

class error_category_impl : public std::error_category {
public:
  const char* name() const noexcept override { return "motis::paxforecast"; }

  std::string message(int ev) const noexcept override {
    switch (ev) {
      case error::ok: return "paxforecast: no error";
      case error::unsupported_measure:
        return "paxforecast: unsupported measure";
      case error::universe_not_found: return "paxforecast: universe not found";
      case error::invalid_rt_update_message:
        return "paxforecast: invalid rt update message";
      default: return "paxforecast: unknown error";
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

}  // namespace motis::paxforecast

namespace std {

template <>
struct is_error_code_enum<motis::paxforecast::error::error_code_t>
    : public std::true_type {};

}  // namespace std
