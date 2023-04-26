#pragma once

#include <cstdint>

namespace motis::module {

enum class json_format : std::uint8_t {
  DEFAULT_FLATBUFFERS,
  SINGLE_LINE,
  TYPES_IN_UNIONS,
  CONTENT_ONLY_TYPES_IN_UNIONS
};

constexpr auto const kDefaultOuputJsonFormat =
    json_format::CONTENT_ONLY_TYPES_IN_UNIONS;

}  // namespace motis::module
