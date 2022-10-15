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
      |
      transform([&](auto&& m) -> std::optional<Offset<MetaStation>> {
        try {
          return std::make_optional(CreateMetaStation(
              fbb, sb.get_or_create_station(m.eva_, fbb),
              fbb.CreateVector(utl::to_vec(
                  get_equivalent_stations(m, hrd_meta_stations), [&](auto&& e) {
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

void add_equivalent_stations(
    std::vector<int>& visited, int stationeva,
    std::set<station_meta_data::meta_station> const& hrd_meta_stations) {
  auto const& s = hrd_meta_stations.find({stationeva, {}});
  if (s == hrd_meta_stations.end()) {
    return;
  }
  for (auto const& e : s->equivalent_) {
    if (std::find(begin(visited), end(visited), e) == end(visited)) {
      visited.push_back(e);
      add_equivalent_stations(visited, e, hrd_meta_stations);
    }
  }
}

std::vector<int> get_equivalent_stations(
    station_meta_data::meta_station const& s,
    std::set<station_meta_data::meta_station> const& hrd_meta_stations) {
  auto stations = s.equivalent_;
  for (auto const& station : s.equivalent_) {
    add_equivalent_stations(stations, station, hrd_meta_stations);
  }
  return stations;
}

}  // namespace motis::loader::hrd
