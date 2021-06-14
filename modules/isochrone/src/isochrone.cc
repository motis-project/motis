#include "motis/isochrone/isochrone.h"

#include "motis/hash_set.h"

#include "boost/program_options.hpp"

#include "utl/to_vec.h"

#include "motis/core/common/logging.h"
#include "motis/core/schedule/schedule.h"
#include "motis/module/context/get_schedule.h"
#include "motis/isochrone/build_query.h"

#include "motis/isochrone/search.h"


#include "motis/protocol/Message_generated.h"

using namespace flatbuffers;
using namespace motis::module;

namespace motis::isochrone {


isochrone::isochrone() : module("Isochrone Options", "isochrone") {}

void isochrone::init(motis::module::registry& reg) {

  reg.register_op("/isochrone",
                  [this](msg_ptr const& m) { return list_stations(m); });
}

msg_ptr isochrone::list_stations(msg_ptr const& msg) {
  auto req = motis_content(IsochroneRequest, msg);
  message_creator mc;

  auto const& sched = get_schedule();
  auto query = build_query(sched, req);



  auto res = search::get_connections(query);

  std::vector<Offset<Station>> stations;
  std::vector<long> travel_times;


  for (auto const& station : res.stations_) {
    auto const pos = Position(station->width_, station->length_);
    stations.emplace_back(CreateStation(mc, mc.CreateString(station->eva_nr_),
                                        mc.CreateString(station->name_), &pos));
  }

  mc.create_and_finish(
      MsgContent_IsochroneResponse,
      CreateIsochroneResponse(mc, mc.CreateVector(stations),
                              mc.CreateVector(res.travel_times_))
          .Union());


  return make_msg(mc);
}

}  // namespace motis::isochrone
