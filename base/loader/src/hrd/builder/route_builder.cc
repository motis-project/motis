#include "motis/loader/hrd/builder/route_builder.h"

#include "utl/get_or_create.h"
#include "utl/to_vec.h"

using namespace utl;
using namespace flatbuffers64;

namespace motis::loader::hrd {

Offset<Route> route_builder::get_or_create_route(
    std::vector<hrd_service::stop> const& stops, station_builder& sb,
    FlatBufferBuilder& fbb) {
  auto events =
      utl::to_vec(begin(stops), end(stops), [&](hrd_service::stop const& s) {
        return stop_restrictions{s.eva_num_, s.dep_.in_out_allowed_,
                                 s.arr_.in_out_allowed_};
      });
  return utl::get_or_create(routes_, events, [&]() {
    return CreateRoute(fbb,
                       fbb.CreateVector(utl::to_vec(
                           begin(events), end(events),
                           [&](stop_restrictions const& sr) {
                             return sb.get_or_create_station(sr.eva_num_, fbb);
                           })),
                       fbb.CreateVector(utl::to_vec(
                           begin(events), end(events),
                           [](stop_restrictions const& sr) -> uint8_t {
                             return sr.entering_allowed_ ? 1 : 0;
                           })),
                       fbb.CreateVector(utl::to_vec(
                           begin(events), end(events),
                           [](stop_restrictions const& sr) -> uint8_t {
                             return sr.leaving_allowed_ ? 1 : 0;
                           })));
  });
}

}  // namespace motis::loader::hrd
