#pragma once

#include <system_error>

namespace motis::path {

namespace error {
enum error_code_t {
  ok = 0,
  database_error = 1,
  not_found = 2,
  unknown_sequence = 3,
  database_not_available = 4,
  invalid_request = 5,
};
}  // namespace error

class error_category_impl : public std::error_category {
public:
  const char* name() const noexcept override;
  std::string message(int ev) const noexcept override;
};

inline const std::error_category& error_category() {
  static error_category_impl instance;
  return instance;
}

namespace error {
std::error_code make_error_code(error_code_t e) noexcept;
}  // namespace error

}  // namespace motis::path

namespace std {

template <>
struct is_error_code_enum<motis::path::error::error_code_t>
    : public std::true_type {};

}  // namespace std
