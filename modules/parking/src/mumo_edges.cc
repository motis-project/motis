#include "utl/to_vec.h"

#include "motis/core/common/constants.h"
#include "motis/module/context/motis_call.h"
#include "motis/parking/mumo_edges.h"

using namespace motis::module;
using namespace motis::routing;
using namespace motis::lookup;
using namespace motis::osrm;
using namespace motis::intermodal;
using namespace motis::ppr;
using namespace geo;
using namespace flatbuffers;

namespace motis::parking {

inline geo::latlng to_latlng(Position const* pos) {
  return {pos->lat(), pos->lng()};
}

msg_ptr make_geo_station_request(geo::latlng const& pos, double radius) {
  Position const fbs_position{pos.lat_, pos.lng_};
  message_creator mc;
  mc.create_and_finish(
      MsgContent_LookupGeoStationRequest,
      CreateLookupGeoStationRequest(mc, &fbs_position, 0, radius).Union(),
      "/lookup/geo_station");
  return make_msg(mc);
}

msg_ptr make_osrm_request(geo::latlng const& pos,
                          std::vector<parking_lot> const& destinations,
                          std::string const& profile, SearchDir direction) {
  Position const fbs_position{pos.lat_, pos.lng_};
  auto const many = utl::to_vec(destinations, [](auto const& dest) {
    return Position{dest.location_.lat_, dest.location_.lng_};
  });

  message_creator mc;
  mc.create_and_finish(
      MsgContent_OSRMOneToManyRequest,
      CreateOSRMOneToManyRequest(mc, mc.CreateString(profile), direction,
                                 &fbs_position, mc.CreateVectorOfStructs(many))
          .Union(),
      "/osrm/one_to_many");
  return make_msg(mc);
}

msg_ptr make_ppr_request(latlng const& pos,
                         std::vector<Position> const& destinations,
                         SearchOptions const* search_options, SearchDir dir,
                         bool include_steps, bool include_edges,
                         bool include_path) {
  assert(search_options != nullptr);
  Position const fbs_position{pos.lat_, pos.lng_};

  message_creator mc;
  mc.create_and_finish(
      MsgContent_FootRoutingRequest,
      CreateFootRoutingRequest(
          mc, &fbs_position, mc.CreateVectorOfStructs(destinations),
          motis_copy_table(SearchOptions, mc, search_options), dir,
          include_steps, include_edges, include_path)
          .Union(),
      "/ppr/route");
  return make_msg(mc);
}

msg_ptr make_ppr_request(::ppr::location const& start,
                         std::vector<::ppr::location> const& destinations,
                         std::string const& profile_name,
                         double const duration_limit, SearchDir dir,
                         bool include_steps, bool include_edges,
                         bool include_path) {
  Position const fbs_position{start.lat(), start.lon()};

  message_creator mc;
  mc.create_and_finish(
      MsgContent_FootRoutingRequest,
      CreateFootRoutingRequest(
          mc, &fbs_position,
          mc.CreateVectorOfStructs(utl::to_vec(
              destinations,
              [](auto const& loc) { return Position{loc.lat(), loc.lon()}; })),
          CreateSearchOptions(mc, mc.CreateString(profile_name),
                              duration_limit),
          dir, include_steps, include_edges, include_path)
          .Union(),
      "/ppr/route");
  return make_msg(mc);
}

msg_ptr make_ppr_request(geo::latlng const& pos,
                         Vector<Offset<Station>> const* stations,
                         SearchOptions const* search_options, SearchDir dir,
                         bool include_steps, bool include_edges,
                         bool include_path) {
  return make_ppr_request(
      pos,
      utl::to_vec(*stations,
                  [](auto const& station) { return *station->pos(); }),
      search_options, dir, include_steps, include_edges, include_path);
}

}  // namespace motis::parking
