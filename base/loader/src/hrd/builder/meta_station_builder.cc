#include "motis/loader/hrd/builder/meta_station_builder.h"

#include <optional>

#include "utl/pipes/all.h"
#include "utl/pipes/remove_if.h"
#include "utl/pipes/transform.h"
#include "utl/pipes/vec.h"
#include "utl/to_vec.h"

#include "motis/core/common/logging.h"

using namespace motis::logging;

namespace motis::loader::hrd {

using namespace flatbuffers64;

Offset<Vector<Offset<MetaStation>>> create_meta_stations(
    std::set<station_meta_data::meta_station> const& hrd_meta_stations,
    station_builder& sb, FlatBufferBuilder& fbb) {
  using namespace utl;
  return fbb.CreateVector(
      all(hrd_meta_stations)  //
      | remove_if([&](auto&& m) { return m.equivalent_.empty(); })  //
      | transform([&](station_meta_data::meta_station const& m)
                      -> std::optional<Offset<MetaStation>> {
          try {
            return std::make_optional(CreateMetaStation(
                fbb, sb.get_or_create_station(m.eva_, fbb),
                fbb.CreateVector(utl::to_vec(m.equivalent_, [&](auto&& e) {
                  return sb.get_or_create_station(e, fbb);
                }))));
          } catch (std::exception const& e) {
            LOG(error) << "meta station error: " << e.what();
            return std::nullopt;
          }
        })  //
      | remove_if([](auto&& opt) { return !opt.has_value(); })  //
      | transform([](auto&& opt) { return *opt; })  //
      | vec());
}

}  // namespace motis::loader::hrd
