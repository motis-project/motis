#pragma once

#include <chrono>
#include <limits>
#include <sstream>

#include "flatbuffers/flatbuffers.h"

#include "rapidjson/document.h"

#include "date/date.h"

#include "utl/verify.h"

#include "motis/core/common/unixtime.h"

namespace motis::ris::ribasis {

struct ris_msg_context {
  unixtime timestamp_{};
  unixtime earliest_{std::numeric_limits<unixtime>::max()};
  unixtime latest_{std::numeric_limits<unixtime>::min()};
  flatbuffers::FlatBufferBuilder b_{};
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

inline unixtime get_schedule_timestamp(ris_msg_context& ctx,
                                       rapidjson::Value const& obj,
                                       char const* key) {
  auto const ts = get_timestamp(obj, key);
  ctx.earliest_ = std::min(ctx.earliest_, ts);
  ctx.latest_ = std::max(ctx.latest_, ts);
  return ts;
}

}  // namespace motis::ris::ribasis
