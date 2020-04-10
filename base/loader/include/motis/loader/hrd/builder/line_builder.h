#pragma once

#include <map>

#include "flatbuffers/flatbuffers.h"

#include "utl/parser/cstr.h"

namespace motis::loader::hrd {

struct line_builder {
  flatbuffers64::Offset<flatbuffers64::String> get_or_create_line(
      std::vector<utl::cstr> const&, flatbuffers64::FlatBufferBuilder&);

  std::map<uint64_t, flatbuffers64::Offset<flatbuffers64::String>> fbs_lines_;
};

}  // namespace motis::loader::hrd
