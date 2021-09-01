#include "gtfsrt_test.h"

#include <utility>

#include "motis/loader/loader.h"
#include "motis/ris/gtfs-rt/gtfsrt_parser.h"
#include "motis/ris/ris_message.h"

namespace motis::ris::gtfsrt {

gtfsrt_test::gtfsrt_test(loader::loader_options options)
    : opts_{std::move(options)} {}

void gtfsrt_test::SetUp() { sched_ = load_schedule(opts_); }

std::vector<ris_message> gtfsrt_test::parse_json(std::string const& json) {
  auto bin = json_to_protobuf(json);
  gtfsrt_parser cut{*sched_};
  return cut.parse(std::string_view{bin.c_str(), bin.size()});
}

}  // namespace motis::ris::gtfsrt
