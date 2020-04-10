#include "motis/loader/hrd/builder/direction_builder.h"

#include "utl/get_or_create.h"

#include "utl/verify.h"

#include "motis/loader/hrd/model/hrd_service.h"
#include "motis/loader/hrd/parse_config.h"
#include "motis/loader/util.h"

namespace motis::loader::hrd {

using namespace flatbuffers64;
using namespace utl;

direction_builder::direction_builder(
    std::map<uint64_t, std::string> hrd_directions)
    : hrd_directions_(std::move(hrd_directions)) {}

Offset<Direction> direction_builder::get_or_create_direction(
    std::vector<std::pair<uint64_t, int>> const& directions,
    station_builder& sb, flatbuffers64::FlatBufferBuilder& fbb) {
  if (directions.empty()) {
    return 0;
  } else {
    auto const direction_key = directions[0];
    return utl::get_or_create(fbs_directions_, direction_key.first, [&]() {
      switch (direction_key.second) {
        case hrd_service::EVA_NUMBER: {
          return CreateDirection(
              fbb, sb.get_or_create_station(direction_key.first, fbb));
        }
        case hrd_service::DIRECTION_CODE: {
          auto it = hrd_directions_.find(direction_key.first);
          utl::verify(it != end(hrd_directions_), "missing direction info: {}",
                      direction_key.first);
          return CreateDirection(fbb, 0,
                                 to_fbs_string(fbb, it->second, ENCODING));
        }
        default: assert(false); return Offset<Direction>(0);
      }
    });
  }
}

}  // namespace motis::loader::hrd
