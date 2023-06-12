#pragma once

#include <algorithm>

#include "motis/protocol/Station_generated.h"
#include "geo/latlng.h"


namespace motis::intermodal {

struct minimalistic_station {
  Position pos;
  geo::latlng geo_pos;
  std::string name;
  std::string id;
};

std::vector<minimalistic_station> first_filter(geo::latlng pos, int max_dur, int max_dist,
                  const flatbuffers::Vector<flatbuffers::Offset<Station>> *stations);

}


