#include "gtfsrt_test.h"

#include <utility>

#include "motis/loader/loader.h"
#include "motis/ris/gtfs-rt/common.h"
#include "motis/ris/gtfs-rt/gtfsrt_parser.h"
#include "motis/ris/gtfs-rt/util.h"
#include "motis/ris/ris_message.h"

namespace motis::ris::gtfsrt {

gtfsrt_test::gtfsrt_test(loader::loader_options options)
    : opts_{std::move(options)} {}

void gtfsrt_test::SetUp() {
  sched_ = load_schedule(opts_);
  knowledge_ = std::make_unique<knowledge_context>("", *sched_);
}

std::vector<ris_message> gtfsrt_test::parse_json(
    std::string const& json) const {
  return parse(*knowledge_, true, std::string_view{json_to_protobuf(json)});
}

}  // namespace motis::ris::gtfsrt
