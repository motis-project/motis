#pragma once

#include <system_error>
#include <type_traits>

namespace motis::access {

namespace error {
enum error_code_t {
  ok = 0,
  not_implemented = 1,
  station_not_found = 2,
  service_not_found = 3,
  timestamp_not_in_schedule = 4
};
}  // namespace error

class error_category_impl : public std::error_category {
public:
  const char* name() const noexcept override { return "motis::access"; }

  std::string message(int ev) const noexcept override {
    switch (ev) {
      case error::ok: return "access: no error";
      case error::not_implemented: return "access: not implemented";
      case error::station_not_found: return "access: station not found";
      case error::service_not_found: return "access: service not found";
      case error::timestamp_not_in_schedule:
        return "access: timestamp not in schedule";
      default: return "access: unknown error";
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

}  // namespace motis::access

namespace std {
template <>
struct is_error_code_enum<motis::access::error::error_code_t>
    : public std::true_type {};

}  // namespace std
