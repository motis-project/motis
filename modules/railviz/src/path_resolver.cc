#include "motis/railviz/path_resolver.h"

#include "utl/get_or_create.h"

#include "utl/verify.h"

#include "motis/core/conv/trip_conv.h"
#include "motis/module/context/motis_call.h"
#include "motis/module/message.h"

using namespace motis::module;
using namespace flatbuffers;

namespace motis::railviz {

path_resolver::path_resolver(schedule const& sched, int zoom_level)
    : sched_(sched), zoom_level_(zoom_level), req_count_(0) {}

std::vector<std::vector<double>> path_resolver::get_trip_path(trip const* trp) {
  return utl::get_or_create(trip_cache_, trp->edges_, [&]() {
    ++req_count_;

    message_creator fbb;
    fbb.create_and_finish(MsgContent_PathByTripIdRequest,
                          path::CreatePathByTripIdRequest(
                              fbb, to_fbs(sched_, fbb, trp), zoom_level_, false)
                              .Union(),
                          "/path/by_trip_id");

    using path::PathSeqResponse;
    auto const path_res = motis_call(make_msg(fbb))->val();
    return utl::to_vec(
        *motis_content(PathSeqResponse, path_res)->segments(),
        [](path::Segment const* s) { return utl::to_vec(*s->coordinates()); });
  });
}

std::pair<bool, std::vector<double>> path_resolver::get_segment_path(
    edge const* e) {
  utl::verify(!e->empty(), "non-empty route edge needed");

  return utl::get_or_create(edge_cache_, e, [&]() {
    auto const first_valid_lcon_it = std::find_if(
        begin(e->m_.route_edge_.conns_), end(e->m_.route_edge_.conns_),
        [](light_connection const& lcon) -> bool { return lcon.valid_ != 0U; });
    utl::verify(first_valid_lcon_it != end(e->m_.route_edge_.conns_),
                "no valid light connection found");

    auto const trp = sched_.merged_trips_[first_valid_lcon_it->trips_]->front();
    try {
      auto const it =
          std::find_if(begin(*trp->edges_), end(*trp->edges_),
                       [&](edge const* trp_e) { return trp_e == e; });
      utl::verify(it != end(*trp->edges_), "trip edge data error");

      auto segment =
          get_trip_path(trp).at(std::distance(begin(*trp->edges_), it));
      utl::verify(segment.size() >= 4, "no empty segments allowed");

      return std::make_pair(false, std::move(segment));
    } catch (std::exception const& ex) {
      auto const& from = *sched_.stations_[e->from_->get_station()->id_];
      auto const& to = *sched_.stations_[e->to_->get_station()->id_];
      return std::make_pair(
          true, std::vector<double>(
                    {from.width_, from.length_, to.width_, to.length_}));
    }
  });
}

}  // namespace motis::railviz
