#pragma once

#include <system_error>
#include <type_traits>

namespace motis::intermodal {

namespace error {
enum error_code_t {
  ok = 0,
  unknown_mode = 1,
  no_guess_for_station = 2,
  parking_edge_error = 3
};
}  // namespace error

class error_category_impl : public std::error_category {
public:
  const char* name() const noexcept override { return "motis::intermodal"; }

  std::string message(int ev) const noexcept override {
    switch (ev) {
      case error::ok: return "intermodal: no error";
      case error::unknown_mode: return "intermodal: unknown mode";
      case error::no_guess_for_station:
        return "routing: station could not be guessed";
      case error::parking_edge_error: return "intermodal: parking edge error";
      default: return "intermodal: unknown error";
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

}  // namespace motis::intermodal

namespace std {

template <>
struct is_error_code_enum<motis::intermodal::error::error_code_t>
    : public std::true_type {};

}  // namespace std
