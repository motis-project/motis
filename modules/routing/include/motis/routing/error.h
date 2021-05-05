#pragma once

#include <system_error>
#include <type_traits>

namespace motis::routing {

namespace error {
enum error_code_t {
  ok = 0,
  no_guess_for_station = 1,
  search_type_not_supported = 2,
  path_length_not_supported = 3,
  journey_date_not_in_schedule = 4,
  event_not_found = 5,
  edge_type_not_supported = 6,
  too_many_start_labels = 7,
  include_equivalent_not_supported = 8
};
}  // namespace error

class error_category_impl : public std::error_category {
public:
  const char* name() const noexcept override { return "motis::routing"; }

  std::string message(int ev) const noexcept override {
    switch (ev) {
      case error::ok: return "routing: no error";
      case error::no_guess_for_station:
        return "routing: station could not be guessed";
      case error::search_type_not_supported:
        return "routing: requested search type not supported";
      case error::path_length_not_supported:
        return "routing: path length not supported";
      case error::journey_date_not_in_schedule:
        return "routing: journey date not in schedule";
      case error::event_not_found: return "routing: event not found";
      case error::edge_type_not_supported:
        return "routing: edge type not supported";
      case error::too_many_start_labels:
        return "routing: too many start labels (route edge not sorted?)";
      case error::include_equivalent_not_supported:
        return "routing: include equivalent not supported";
      default: return "routing: unknown error";
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

}  // namespace motis::routing

namespace std {

template <>
struct is_error_code_enum<motis::routing::error::error_code_t>
    : public std::true_type {};

}  // namespace std
