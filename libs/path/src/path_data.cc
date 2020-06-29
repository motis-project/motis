#include "motis/path/path_data.h"

#include "motis/core/access/trip_iterator.h"

#include "motis/path/path_database_query.h"

namespace mm = motis::module;
namespace ma = motis::access;

namespace motis::path {

size_t path_data::trip_to_index(schedule const& sched, trip const* trp) const {
  utl::verify(ma::sections::begin(trp) != ma::sections::end(trp),
              "trip_to_index: invalid trip");

  return index_->find(
      {utl::to_vec(access::stops(trp),
                   [&](auto const& stop) {
                     return stop.get_station(sched).eva_nr_.str();
                   }),
       (*std::min_element(ma::sections::begin(trp), ma::sections::end(trp),
                          [](auto const& lhs, auto const& rhs) {
                            return lhs.fcon().clasz_ < rhs.fcon().clasz_;
                          }))
           .fcon()
           .clasz_});
}

mm::msg_ptr path_data::get_response(size_t const index,
                                    int const zoom_level) const {
  mm::message_creator mc;
  mc.create_and_finish(MsgContent_PathSeqResponse,
                       reconstruct_sequence(mc, index, zoom_level).Union());
  return make_msg(mc);
}

flatbuffers::Offset<PathSeqResponse> path_data::reconstruct_sequence(
    mm::message_creator& mc, size_t const index, int const zoom_level) const {
  path_database_query q{zoom_level};
  q.add_sequence(index);
  q.execute(*db_);
  return q.write_sequence(mc, *db_, index);
}

}  // namespace motis::path
