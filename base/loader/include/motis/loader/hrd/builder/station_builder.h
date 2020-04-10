#pragma once

#include <map>

#include "motis/loader/hrd/model/timezones.h"
#include "motis/loader/hrd/parser/stations_parser.h"

#include "motis/schedule-format/Station_generated.h"

namespace motis::loader::hrd {

struct station_builder {
  station_builder(std::map<int, intermediate_station>, timezones);

  flatbuffers64::Offset<Station> get_or_create_station(
      int, flatbuffers64::FlatBufferBuilder&);

  std::map<int, intermediate_station> hrd_stations_;
  timezones timezones_;
  std::map<int, flatbuffers64::Offset<Station>> fbs_stations_;
  std::map<timezone_entry const*, flatbuffers64::Offset<Timezone>>
      fbs_timezones_;
};

}  // namespace motis::loader::hrd
