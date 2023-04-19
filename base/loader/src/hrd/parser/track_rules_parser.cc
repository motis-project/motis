#include "motis/loader/hrd/parser/track_rules_parser.h"

#include "utl/parser/arg_parser.h"
#include "utl/parser/cstr.h"

#include "motis/core/common/date_time_util.h"
#include "motis/core/common/logging.h"
#include "motis/loader/hrd/parser/bitfields_parser.h"
#include "motis/loader/parser_error.h"
#include "motis/loader/util.h"

using namespace utl;
using namespace flatbuffers64;
using namespace motis::logging;

namespace motis::loader::hrd {

track_rules parse_track_rules(loaded_file const& file,
                              flatbuffers64::FlatBufferBuilder& b,
                              config const& c) {
  scoped_timer const timer("parsing track rules");
  track_rules prs;
  std::map<uint64_t, Offset<String>> track_names;

  for_each_line_numbered(file.content(), [&](cstr line, int line_number) {
    if (line.len == 0 || line.starts_with("%")) {
      return;
    } else if (line.len < c.track_rul_.min_line_length_) {
      throw parser_error(file.name(), line_number);
    }

    auto eva_num = parse<int>(line.substr(c.track_rul_.eva_num_));
    auto train_num = parse<int>(line.substr(c.track_rul_.train_num_));
    auto train_admin =
        raw_to_int<uint64_t>(line.substr(c.track_rul_.train_admin_));
    auto track_name_str = line.substr(c.track_rul_.track_name_).trim();
    auto track_name = raw_to_int<uint64_t>(track_name_str);
    auto time = hhmm_to_min(
        parse<int>(line.substr(c.track_rul_.time_).trim(), TIME_NOT_SET));
    auto bitfield =
        parse<int>(line.substr(c.track_rul_.bitfield_).trim(), ALL_DAYS_KEY);

    // Resolve track name (create it if not found)
    auto track_name_it = track_names.find(track_name);
    if (track_name_it == end(track_names)) {
      std::tie(track_name_it, std::ignore) = track_names.insert(std::make_pair(
          track_name, to_fbs_string(b, track_name_str, "ISO-8859-1")));
    }

    prs[std::make_tuple(eva_num, train_num, train_admin)].push_back(
        {track_name_it->second, bitfield, time});
  });
  return prs;
}

}  // namespace motis::loader::hrd
