#pragma once

#include <system_error>
#include <type_traits>

namespace motis::module {

namespace error {
enum error_code_t {
  ok = 0,
  unable_to_parse_msg = 1,
  malformed_msg = 2,
  target_not_found = 3,
  unknown_error = 4,
  unexpected_message_type = 5,
  null_message_content_access = 6,
  remote_error = 7
};
}  // namespace error

class error_category_impl : public std::error_category {
public:
  const char* name() const noexcept override { return "motis::module"; }

  std::string message(int ev) const noexcept override {
    switch (ev) {
      case error::ok: return "module: no error";
      case error::unable_to_parse_msg: return "module: unable to parse message";
      case error::malformed_msg: return "module: malformed message";
      case error::target_not_found: return "module: target not found";
      case error::unexpected_message_type:
        return "module: unexpected message type";
      case error::remote_error: return "module: remote execution error";
      case error::unknown_error:
      default: return "module: unknown error";
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

}  // namespace motis::module

namespace std {
template <>
struct is_error_code_enum<motis::module::error::error_code_t>
    : public std::true_type {};
}  // namespace std
