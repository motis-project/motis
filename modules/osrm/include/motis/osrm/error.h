#pragma once

#include <system_error>
#include <type_traits>

namespace motis::osrm {

namespace error {
enum error_code_t {
  ok = 0,
  profile_not_available = 1,
  no_routing_response = 2
};
}  // namespace error

class error_category_impl : public std::error_category {
public:
  const char* name() const noexcept override { return "motis::osrm"; }

  std::string message(int ev) const noexcept override {
    switch (ev) {
      case error::profile_not_available: return "osrm: profile not available";
      case error::no_routing_response: return "osrm: no routing response";
      default: return "osrm: unknown error";
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
}  // namespace motis::osrm

namespace std {

template <>
struct is_error_code_enum<motis::osrm::error::error_code_t>
    : public std::true_type {};

}  // namespace std
