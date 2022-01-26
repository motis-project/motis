#pragma once

#include <charconv>
#include <string_view>

#include "rapidjson/document.h"

#include "utl/verify.h"

namespace motis::json {

rapidjson::Value const& get_value(rapidjson::Value const& parent,
                                  char const* key);

rapidjson::Value const& get_obj(rapidjson::Value const& parent,
                                char const* key);

rapidjson::Value::ConstArray get_array(rapidjson::Value const& obj,
                                       char const* key);

std::string_view get_str(rapidjson::Value const& obj, char const* key);

std::string_view get_optional_str(rapidjson::Value const& obj, char const* key);

bool get_bool(rapidjson::Value const& obj, char const* key);

template <typename T>
T get_parsed_number(rapidjson::Value const& obj, char const* key) {
  auto val = T{};
  auto const s = get_str(obj, key);
  auto const result = std::from_chars(s.data(), s.data() + s.size(), val);
  utl::verify(result.ec == std::errc{} && result.ptr == s.data() + s.size(),
              "not a number ({}): {}", key, s);
  return val;
}

}  // namespace motis::json
