#pragma once

#include <system_error>
#include <type_traits>

namespace motis::paxmon {

namespace error {
enum error_code_t {
  ok = 0,
  universe_not_found = 1,
  universe_destruction_failed = 2
};
}  // namespace error

class error_category_impl : public std::error_category {
public:
  const char* name() const noexcept override { return "motis::paxmon"; }

  std::string message(int ev) const noexcept override {
    switch (ev) {
      case error::ok: return "paxmon: no error";
      case error::universe_not_found: return "paxmon: universe not found";
      case error::universe_destruction_failed:
        return "paxmon: universe destruction failed";
      default: return "paxmon: unknown error";
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

}  // namespace motis::paxmon

namespace std {

template <>
struct is_error_code_enum<motis::paxmon::error::error_code_t>
    : public std::true_type {};

}  // namespace std
