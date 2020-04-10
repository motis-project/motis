#pragma once

#include <map>
#include <vector>

#include "motis/loader/hrd/builder/station_builder.h"
#include "motis/loader/hrd/parser/station_meta_data_parser.h"

#include "motis/schedule-format/Footpath_generated.h"

namespace motis::loader::hrd {

flatbuffers64::Offset<flatbuffers64::Vector<flatbuffers64::Offset<Footpath>>>
create_footpaths(std::set<station_meta_data::footpath> const&, station_builder&,
                 flatbuffers64::FlatBufferBuilder&);

}  // namespace motis::loader::hrd
