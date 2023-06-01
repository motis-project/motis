#pragma once

#include <system_error>
#include <type_traits>

namespace motis::rt {

namespace error {
enum error_code_t {
  ok = 0,
  sanity_check_failed = 1,
  schedule_not_found = 2,
};
}  // namespace error

class error_category_impl : public std::error_category {
public:
  const char* name() const noexcept override { return "motis::rt"; }

  std::string message(int ev) const noexcept override {
    switch (ev) {
      case error::ok: return "rt: no error";
      case error::sanity_check_failed: return "rt: sanity check failed";
      case error::schedule_not_found: return "rt: schedule not found";
      default: return "rt: unkown error";
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

}  // namespace motis::rt

namespace std {
template <>
struct is_error_code_enum<motis::rt::error::error_code_t>
    : public std::true_type {};

}  // namespace std
