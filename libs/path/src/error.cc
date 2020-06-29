#include "motis/path/error.h"

namespace motis::path {

const char* error_category_impl::name() const noexcept { return "motis::path"; }

std::string error_category_impl::message(int ev) const noexcept {
  switch (ev) {
    case error::ok: return "path: no error";
    case error::database_error: return "path: database error";
    case error::not_found: return "path: not found";
    case error::unknown_sequence: return "path: unknown sequence";
    case error::database_not_available: return "path: database not available";
    case error::invalid_request: return "path: invalid request";
    default: return "path: unkown error";
  }
}

namespace error {
std::error_code make_error_code(error_code_t e) noexcept {
  return std::error_code(static_cast<int>(e), error_category());
}
}  // namespace error

}  // namespace motis::path
