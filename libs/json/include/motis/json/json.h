#pragma once

#include <charconv>
#include <cstdint>
#include <string_view>

#include "boost/uuid/uuid.hpp"

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

int get_int(rapidjson::Value const& obj, char const* key);

unsigned get_uint(rapidjson::Value const& obj, char const* key);

std::int64_t get_int64(rapidjson::Value const& obj, char const* key);

std::uint64_t get_uint64(rapidjson::Value const& obj, char const* key);

double get_double(rapidjson::Value const& obj, char const* key);

template <typename T>
T get_parsed_number(rapidjson::Value const& obj, char const* key) {
  auto val = T{};
  auto const s = get_str(obj, key);
  auto const result = std::from_chars(s.data(), s.data() + s.size(), val);
  utl::verify(result.ec == std::errc{} && result.ptr == s.data() + s.size(),
              "not a number ({}): {}", key, s);
  return val;
}

boost::uuids::uuid get_uuid(rapidjson::Value const& obj, char const* key);

}  // namespace motis::json
