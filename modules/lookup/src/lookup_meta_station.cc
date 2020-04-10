#include "motis/lookup/lookup_meta_station.h"

#include "motis/core/access/station_access.h"
#include "motis/lookup/util.h"

using namespace flatbuffers;

namespace motis::lookup {

Offset<LookupMetaStationResponse> lookup_meta_station(
    FlatBufferBuilder& fbb, schedule const& sched,
    LookupMetaStationRequest const* req) {
  std::vector<Offset<Station>> equivalent;

  auto station = get_station(sched, req->station_id()->str());
  for (auto const& e : station->equivalent_) {
    equivalent.push_back(create_station(fbb, *e));
  }

  return CreateLookupMetaStationResponse(fbb, fbb.CreateVector(equivalent));
}

}  // namespace motis::lookup
