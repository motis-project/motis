#pragma once

#include <map>
#include <vector>

#include "motis/loader/hrd/builder/station_builder.h"
#include "motis/loader/hrd/parser/station_meta_data_parser.h"

#include "motis/schedule-format/MetaStation_generated.h"

namespace motis::loader::hrd {

flatbuffers64::Offset<flatbuffers64::Vector<flatbuffers64::Offset<MetaStation>>>
create_meta_stations(std::set<station_meta_data::meta_station> const&,
                     station_builder& sb, flatbuffers64::FlatBufferBuilder&);

}  // namespace motis::loader::hrd
