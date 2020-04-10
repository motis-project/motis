#pragma once

#include <cinttypes>
#include <vector>

#include "motis/schedule-format/Service_generated.h"

#include "motis/loader/hrd/builder/attribute_builder.h"
#include "motis/loader/hrd/builder/bitfield_builder.h"
#include "motis/loader/hrd/builder/category_builder.h"
#include "motis/loader/hrd/builder/direction_builder.h"
#include "motis/loader/hrd/builder/line_builder.h"
#include "motis/loader/hrd/builder/provider_builder.h"
#include "motis/loader/hrd/builder/route_builder.h"
#include "motis/loader/hrd/builder/station_builder.h"
#include "motis/loader/hrd/model/hrd_service.h"
#include "motis/loader/hrd/parser/track_rules_parser.h"

namespace motis::loader::hrd {

struct service_builder {
  explicit service_builder(track_rules);

  flatbuffers64::Offset<Service> create_service(
      hrd_service const&, route_builder&, station_builder&, category_builder&,
      provider_builder&, line_builder&, attribute_builder&, bitfield_builder&,
      direction_builder&, flatbuffers64::FlatBufferBuilder&,
      bool is_rule_participant);

  track_rules track_rules_;
  std::vector<flatbuffers64::Offset<Service>> fbs_services_;
  std::map<char const*, flatbuffers64::Offset<flatbuffers64::String>>
      filenames_;
};

}  // namespace motis::loader::hrd
