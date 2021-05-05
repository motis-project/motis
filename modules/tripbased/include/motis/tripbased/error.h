#pragma once

#include <system_error>
#include <type_traits>

namespace motis::tripbased {

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
  invalid_additional_edges = 8,
  trip_not_found = 9,
  include_equivalent_not_supported = 10
};
}  // namespace error

class error_category_impl : public std::error_category {
public:
  const char* name() const noexcept override { return "motis::tripbased"; }

  std::string message(int ev) const noexcept override {
    switch (ev) {
      case error::ok: return "tripbased: no error";
      case error::not_implemented: return "tripbased: not implemented";
      case error::internal_error: return "tripbased: internal error";
      case error::no_guess_for_station:
        return "tripbased: station could not be guessed";
      case error::search_type_not_supported:
        return "tripbased: requested search type not supported";
      case error::journey_date_not_in_schedule:
        return "tripbased: journey date not in schedule";
      case error::start_type_not_supported:
        return "tripbased: start type not supported";
      case error::via_not_supported: return "tripbased: via not supported";
      case error::invalid_additional_edges:
        return "tripbased: invalid additional edges";
      case error::trip_not_found: return "tripbased: trip not found";
      case error::include_equivalent_not_supported:
        return "tripbased: include equivalent not supported";
      default: return "tripbased: unknown error";
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
}  // namespace motis::tripbased

namespace std {

template <>
struct is_error_code_enum<motis::tripbased::error::error_code_t>
    : public std::true_type {};

}  // namespace std
