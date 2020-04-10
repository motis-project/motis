#include "motis/loader/hrd/builder/footpath_builder.h"

#include "motis/core/common/logging.h"

namespace motis::loader::hrd {

using namespace flatbuffers64;

Offset<Vector<Offset<Footpath>>> create_footpaths(
    std::set<station_meta_data::footpath> const& hrd_footpaths,
    station_builder& stb, FlatBufferBuilder& fbb) {
  std::vector<Offset<Footpath>> fbs_footpaths;
  for (auto const& f : hrd_footpaths) {
    try {
      fbs_footpaths.push_back(
          CreateFootpath(fbb,  //
                         stb.get_or_create_station(f.from_eva_num_, fbb),
                         stb.get_or_create_station(f.to_eva_num_, fbb),  //
                         f.duration_));
    } catch (std::runtime_error const&) {
      LOG(logging::error) << "skipping footpath " << f.from_eva_num_ << " -> "
                          << f.to_eva_num_
                          << ", could not resolve station(s)\n";
    }
  }
  return fbb.CreateVector(fbs_footpaths);
}

}  // namespace motis::loader::hrd
