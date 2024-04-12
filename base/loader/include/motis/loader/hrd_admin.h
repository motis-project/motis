#pragma once

#include <cstdint>
#include <cstring>
#include <algorithm>
#include <string_view>

namespace motis::loader {

inline std::uint64_t hrd_admin_str_to_int(std::string_view const str) {
  auto result = std::uint64_t{0};
  auto len = str.find('_');
  if (len == std::string_view::npos) {
    len = str.length();
  }
  std::memcpy(&result, str.data(), std::min(len, sizeof result));
  return result;
}

template <typename StrType = std::string>
inline StrType hrd_admin_int_to_str(std::uint64_t const int_admin) {
  char str[9] = {};
  std::memcpy(str, &int_admin, sizeof(int_admin));
  return {str};
}

}  // namespace motis::loader
