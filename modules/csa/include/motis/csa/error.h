#pragma once

#include <system_error>
#include <type_traits>

namespace motis::csa {

namespace error {
enum error_code_t {
  ok = 0,
  not_implemented = 1,
  internal_error = 2,
  no_guess_for_station = 3,
  search_type_not_supported = 4,
  journey_date_not_in_schedule = 5,
  start_type_not_supported = 6,
  via_not_supported = 7,
  additional_edges_not_supported = 8,
  trip_not_found = 9,
  start_footpaths_no_disable = 10,
  include_equivalent_not_supported = 11
};
}  // namespace error

class error_category_impl : public std::error_category {
public:
  const char* name() const noexcept override { return "motis::csa"; }

  std::string message(int ev) const noexcept override {
    switch (ev) {
      case error::ok: return "csa: no error";
      case error::not_implemented: return "csa: not implemented";
      case error::internal_error: return "csa: internal error";
      case error::no_guess_for_station:
        return "csa: station could not be guessed";
      case error::search_type_not_supported:
        return "csa: requested search type not supported";
      case error::journey_date_not_in_schedule:
        return "csa: journey date not in schedule";
      case error::start_type_not_supported:
        return "csa: start type not supported";
      case error::via_not_supported: return "csa: via not supported";
      case error::additional_edges_not_supported:
        return "csa: additional edges not supported";
      case error::trip_not_found: return "csa: trip not found";
      case error::start_footpaths_no_disable:
        return "csa: start footpaths cannot be disabled";
      case error::include_equivalent_not_supported:
        return "csa: include equivalent not supported";
      default: return "csa: unknown error";
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
}  // namespace motis::csa

namespace std {

template <>
struct is_error_code_enum<motis::csa::error::error_code_t>
    : public std::true_type {};

}  // namespace std
