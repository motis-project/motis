#pragma once

#include <system_error>
#include <type_traits>

namespace motis::ppr {

namespace error {
enum error_code_t { ok = 0, profile_not_available = 1 };
}  // namespace error

class error_category_impl : public std::error_category {
public:
  const char* name() const noexcept override { return "motis::ppr"; }

  std::string message(int ev) const noexcept override {
    switch (ev) {
      case error::ok: return "ppr: no error";
      case error::profile_not_available: return "ppr: profile not available";
      default: return "ppr: unknown error";
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
}  // namespace motis::ppr

namespace std {

template <>
struct is_error_code_enum<motis::ppr::error::error_code_t>
    : public std::true_type {};

}  // namespace std
