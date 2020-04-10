#include "motis/loader/hrd/builder/line_builder.h"

#include "utl/get_or_create.h"

#include "motis/loader/util.h"

namespace motis::loader::hrd {

using namespace flatbuffers64;
using namespace utl;

Offset<String> line_builder::get_or_create_line(
    std::vector<utl::cstr> const& lines, FlatBufferBuilder& fbb) {
  if (lines.empty()) {
    return 0;
  } else {
    return utl::get_or_create(
        fbs_lines_, raw_to_int<uint64_t>(lines[0]),
        [&]() { return to_fbs_string(fbb, lines[0], "ISO-8859-1"); });
  }
}

}  // namespace motis::loader::hrd
