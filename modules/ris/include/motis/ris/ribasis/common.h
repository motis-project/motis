#pragma once

#include <charconv>
#include <chrono>
#include <ctime>
#include <algorithm>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>

#include "flatbuffers/flatbuffers.h"

#include "rapidjson/document.h"

#include "date/date.h"

#include "utl/verify.h"

#include "motis/hash_map.h"
#include "motis/protocol/RISMessage_generated.h"

namespace motis::ris::ribasis {

struct context {
  explicit context(std::time_t timestamp)
      : timestamp_{timestamp},
        earliest_{std::numeric_limits<std::time_t>::max()},
        latest_{std::numeric_limits<std::time_t>::min()} {}

  flatbuffers::FlatBufferBuilder b_{};
  std::time_t timestamp_, earliest_, latest_;

  mcd::hash_map<std::string, flatbuffers::Offset<StationInfo>> stations_;
  mcd::hash_map<std::string, flatbuffers::Offset<CategoryInfo>> categories_;
  mcd::hash_map<std::string, flatbuffers::Offset<flatbuffers::String>> lines_;
  mcd::hash_map<std::string, flatbuffers::Offset<ProviderInfo>> providers_;
};

inline rapidjson::Value const& get_value(rapidjson::Value const& parent,
                                         char const* key) {
  auto const member = parent.FindMember(key);
  utl::verify(member != parent.MemberEnd(), "missing key: {}", key);
  return member->value;
}

inline rapidjson::Value const& get_obj(rapidjson::Value const& parent,
                                       char const* key) {
  auto const& value = get_value(parent, key);
  utl::verify(value.IsObject(), "not an object: {}", key);
  return value;
}

inline rapidjson::Value::ConstArray get_array(rapidjson::Value const& obj,
                                              char const* key) {
  auto const& value = get_value(obj, key);
  utl::verify(value.IsArray(), "not a string: {}", key);
  return value.GetArray();
}

inline std::string_view get_str(rapidjson::Value const& obj, char const* key) {
  auto const& value = get_value(obj, key);
  utl::verify(value.IsString(), "not a string: {}", key);
  return {value.GetString(), value.GetStringLength()};
}

inline std::string_view get_optional_str(rapidjson::Value const& obj,
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

inline bool get_bool(rapidjson::Value const& obj, char const* key) {
  auto const& value = get_value(obj, key);
  utl::verify(value.IsBool(), "not a bool: {}", key);
  return value.GetBool();
}

template <typename T>
T get_parsed_number(rapidjson::Value const& obj, char const* key) {
  auto val = T{};
  auto const s = get_str(obj, key);
  auto const result = std::from_chars(s.data(), s.data() + s.size(), val);
  utl::verify(result.ec == std::errc{} && result.ptr == s.data() + s.size(),
              "not a number ({}): {}", key, s);
  return val;
}

inline std::time_t get_timestamp(rapidjson::Value const& obj, char const* key,
                          char const* format = "%FT%T%Ez") {
  auto const s = get_str(obj, key);
  auto tp = date::sys_time<std::chrono::seconds>{};
  auto ss = std::stringstream{};
  ss << s;
  ss >> date::parse(format, tp);
  utl::verify(!ss.fail(), "could not parse timestamp ({}): {}", key, s);
  return tp.time_since_epoch().count();
}

inline std::time_t get_schedule_timestamp(context& ctx, rapidjson::Value const& obj,
                                   char const* key) {
  auto const ts = get_timestamp(obj, key);
  ctx.earliest_ = std::min(ctx.earliest_, ts);
  ctx.latest_ = std::max(ctx.latest_, ts);
  return ts;
}

}  // namespace motis::ris::ribasis
