#include "motis/json/json.h"

namespace motis::json {

rapidjson::Value const& get_value(rapidjson::Value const& parent,
                                  char const* key) {
  auto const member = parent.FindMember(key);
  utl::verify(member != parent.MemberEnd(), "missing key: {}", key);
  return member->value;
}

rapidjson::Value const& get_obj(rapidjson::Value const& parent,
                                char const* key) {
  auto const& value = get_value(parent, key);
  utl::verify(value.IsObject(), "not an object: {}", key);
  return value;
}

rapidjson::Value::ConstArray get_array(rapidjson::Value const& obj,
                                       char const* key) {
  auto const& value = get_value(obj, key);
  utl::verify(value.IsArray(), "not a string: {}", key);
  return value.GetArray();
}

std::string_view get_str(rapidjson::Value const& obj, char const* key) {
  auto const& value = get_value(obj, key);
  utl::verify(value.IsString(), "not a string: {}", key);
  return {value.GetString(), value.GetStringLength()};
}

std::string_view get_optional_str(rapidjson::Value const& obj,
                                  char const* key) {
  auto const& value = get_value(obj, key);
  if (value.IsString()) {
    return {value.GetString(), value.GetStringLength()};
  } else if (value.IsNull()) {
    return {};
  } else {
    throw utl::fail("not a string or null: {}", key);
  }
}

bool get_bool(rapidjson::Value const& obj, char const* key) {
  auto const& value = get_value(obj, key);
  utl::verify(value.IsBool(), "not a bool: {}", key);
  return value.GetBool();
}

}  // namespace motis::json
