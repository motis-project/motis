#pragma once

#include <string_view>

#include "boost/uuid/string_generator.hpp"

#include "flatbuffers/flatbuffers.h"

namespace motis::rt {

inline std::string_view view(flatbuffers::String const* s) {
  return {s->c_str(), s->size()};
}

inline boost::uuids::uuid parse_uuid(std::string_view const sv) {
  return boost::uuids::string_generator{}(sv.begin(), sv.end());
}

}  // namespace motis::rt
