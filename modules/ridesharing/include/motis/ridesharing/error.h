#pragma once

#include "boost/type_traits.hpp"

#include <system_error>

namespace motis::ridesharing {

namespace error {
enum error_code_t {
  ok = 0,
  not_implemented = 1,
  not_initialized = 2,
  database_error = 3,
  database_not_initialized = 4,
  lift_not_found = 5,
  search_failure = 6,
  init_error = 7
};
}  // namespace error

class error_category_impl : public std::error_category {
public:
  const char* name() const noexcept override { return "motis::ridesharing"; }

  std::string message(int ev) const noexcept override {
    switch (ev) {
      case error::ok: return "ridesharing: no error";
      case error::not_implemented: return "ridesharing: not implemented";
      case error::not_initialized: return "ridesharing: not initialized";
      case error::database_error: return "ridesharing: database error";
      case error::database_not_initialized:
        return "ridesharing: database not initialized";
      case error::lift_not_found: return "ridesharing: lift not found";
      case error::search_failure: return "ridesharing: search_failure";
      case error::init_error: return "ridesharing: init error";
      default: return "ridesharing: unkown error";
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
}  // namespace motis::ridesharing

namespace std {

template <>
struct is_error_code_enum<motis::ridesharing::error::error_code_t>
    : public std::true_type {};

}  // namespace std
