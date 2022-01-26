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

#include "motis/core/common/unixtime.h"
#include "motis/hash_map.h"
#include "motis/protocol/RISMessage_generated.h"

namespace motis::ris::ribasis {

struct context {
  explicit context(unixtime timestamp)
      : timestamp_{timestamp},
        earliest_{std::numeric_limits<unixtime>::max()},
        latest_{std::numeric_limits<unixtime>::min()} {}

  flatbuffers::FlatBufferBuilder b_{};
  unixtime timestamp_, earliest_, latest_;

  mcd::hash_map<std::string, flatbuffers::Offset<StationInfo>> stations_;
  mcd::hash_map<std::string, flatbuffers::Offset<CategoryInfo>> categories_;
  mcd::hash_map<std::string, flatbuffers::Offset<flatbuffers::String>> lines_;
  mcd::hash_map<std::string, flatbuffers::Offset<ProviderInfo>> providers_;
};

inline unixtime get_timestamp(rapidjson::Value const& obj, char const* key,
                              char const* format = "%FT%T%Ez") {
  using namespace motis::json;
  auto const s = get_str(obj, key);
  auto tp = date::sys_time<std::chrono::nanoseconds>{};
  auto ss = std::stringstream{};
  ss << s;
  ss >> date::parse(format, tp);
  utl::verify(!ss.fail(), "could not parse timestamp ({}): {}", key, s);
  return std::chrono::duration_cast<std::chrono::seconds>(tp.time_since_epoch())
      .count();
}

inline unixtime get_schedule_timestamp(context& ctx,
                                       rapidjson::Value const& obj,
                                       char const* key) {
  auto const ts = get_timestamp(obj, key);
  ctx.earliest_ = std::min(ctx.earliest_, ts);
  ctx.latest_ = std::max(ctx.latest_, ts);
  return ts;
}

}  // namespace motis::ris::ribasis
