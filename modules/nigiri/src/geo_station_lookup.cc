#include "motis/nigiri/geo_station_lookup.h"

#include "nigiri/timetable.h"

#include "utl/to_vec.h"

#include "motis/core/conv/position_conv.h"
#include "motis/nigiri/location.h"

namespace mm = motis::module;

namespace motis::nigiri {

motis::module::msg_ptr geo_station_lookup(station_lookup const& index,
                                          motis::module::msg_ptr const& msg) {
  using motis::lookup::CreateLookupGeoStationRequest;
  using motis::lookup::CreateLookupGeoStationResponse;
  using motis::lookup::LookupGeoStationRequest;
  auto const req = motis_content(LookupGeoStationRequest, msg);
  mm::message_creator mc;
  mc.create_and_finish(
      MsgContent_LookupGeoStationResponse,
      CreateLookupGeoStationResponse(
          mc, mc.CreateVector(utl::to_vec(
                  index.in_radius({req->pos()->lat(), req->pos()->lng()},
                                  req->min_radius(), req->max_radius()),
                  [&](auto const idx) {
                    auto const station = index.get(idx);
                    auto const pos = to_fbs(station.pos_);
                    return CreateStation(
                        mc,
                        mc.CreateString(
                            fmt::format("{}{}", station.tag_, station.id_)),
                        mc.CreateString(station.id_), &pos);
                  })))
          .Union());
  return mm::make_msg(mc);
}

motis::module::msg_ptr station_location(tag_lookup const& tags,
                                        ::nigiri::timetable const& tt,
                                        motis::module::msg_ptr const& msg) {
  using motis::lookup::CreateLookupStationLocationResponse;
  using motis::lookup::LookupStationLocationResponse;
  using routing::InputStation;
  auto const req = motis_content(InputStation, msg);
  auto const l_idx = get_location_idx(tags, tt, req->id()->view());
  auto const pos = to_fbs(tt.locations_.coordinates_.at(l_idx));
  mm::message_creator mc;
  mc.create_and_finish(MsgContent_LookupStationLocationResponse,
                       CreateLookupStationLocationResponse(mc, &pos).Union());
  return mm::make_msg(mc);
}

}  // namespace motis::nigiri