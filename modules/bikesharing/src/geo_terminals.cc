#include "motis/bikesharing/geo_terminals.h"

#include <iostream>
#include <vector>

#include "motis/module/message.h"

using namespace flatbuffers;
using namespace motis;
using namespace motis::module;

namespace motis::bikesharing {

module::msg_ptr geo_terminals(database const& db, geo_index const& index,
                              BikesharingGeoTerminalsRequest const* req) {
  // TODO(Sebastian Fahnenschreiber) adjust by actual walk distance to the
  // terminal
  auto bucket = timestamp_to_bucket(req->timestamp());

  message_creator mc;
  std::vector<Offset<AvailableBikesharingTerminal>> result;
  for (auto&& t : index.get_terminals(req->pos()->lat(), req->pos()->lng(),
                                      req->radius())) {
    auto terminal = db.get(t.id_);

    Position pos = {terminal.get()->lat(), terminal.get()->lng()};
    auto const availability =
        get_availability(terminal.get()->availability()->Get(bucket),
                         req->availability_aggregator());
    result.push_back(CreateAvailableBikesharingTerminal(
        mc, mc.CreateString(t.id_), &pos, availability));
  }

  mc.create_and_finish(
      MsgContent_BikesharingGeoTerminalsResponse,
      CreateBikesharingGeoTerminalsResponse(mc, mc.CreateVector(result))
          .Union());
  return make_msg(mc);
}

}  // namespace motis::bikesharing
