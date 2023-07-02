#include "motis/core/debug/fbs.h"

#include "flatbuffers/flatbuffers.h"

#include "motis/core/conv/trip_conv.h"

#include "motis/module/message.h"

namespace motis::debug {

std::string to_fbs_json(schedule const& sched, motis::trip const* trp,
                        motis::module::json_format const jf) {
  flatbuffers::FlatBufferBuilder fbb;
  fbb.Finish(to_fbs(sched, fbb, trp));
  return motis::module::fbs_table_to_json(
      flatbuffers::GetRoot<TripId>(fbb.GetBufferPointer()), "motis.TripId", jf);
}

}  // namespace motis::debug
