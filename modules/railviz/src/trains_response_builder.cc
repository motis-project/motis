#include "motis/railviz/trains_response_builder.h"

#include "utl/concat.h"
#include "utl/equal_ranges_linear.h"
#include "utl/erase_duplicates.h"
#include "utl/get_or_create.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "motis/core/access/bfs.h"
#include "motis/core/access/realtime_access.h"
#include "motis/core/access/service_access.h"
#include "motis/core/access/trip_iterator.h"

#include "motis/core/conv/station_conv.h"
#include "motis/core/conv/timestamp_reason_conv.h"
#include "motis/core/conv/trip_conv.h"

#include "motis/module/context/motis_call.h"

using namespace motis::access;
using namespace motis::logging;
using namespace motis::module;
using namespace flatbuffers;

namespace motis::railviz {

void trains_response_builder::add_train_full(ev_key k) {
  for (auto const& e : route_bfs(k, bfs_direction::BOTH, false)) {
    add_ev_key(ev_key{e, k.lcon_idx_, event_type::DEP});
  }
}

void trains_response_builder::add_ev_key(ev_key k) {
  auto const& trp = sched_.merged_trips_.at(k.lcon()->trips_)->at(0);

  auto const it = std::find_if(
      sections::begin(trp), sections::end(trp),
      [&](auto const& s) { return s.edge_ == k.route_edge_.get_edge(); });
  utl::verify(it != sections::end(trp),
              "trains_response_builder: missing edge");

  queries_.emplace_back(trp, std::distance(sections::begin(trp), it));
}

msg_ptr trains_response_builder::resolve_paths() {
  utl::erase_duplicates(queries_);  // this sorts!

  message_creator mc;
  std::vector<Offset<path::TripIdSegments>> trip_segments;
  utl::equal_ranges_linear(
      queries_,
      [](auto const& lhs, auto const& rhs) { return lhs.first == rhs.first; },
      [&](auto lb, auto ub) {
        trip_segments.emplace_back(path::CreateTripIdSegments(
            mc, to_fbs(sched_, mc, lb->first),
            mc.CreateVector(utl::to_vec(
                lb, ub, [](auto const& q) -> uint32_t { return q.second; }))));
      });

  mc.create_and_finish(MsgContent_PathByTripIdBatchRequest,
                       CreatePathByTripIdBatchRequest(
                           mc, mc.CreateVector(trip_segments), zoom_level_)
                           .Union(),
                       "/path/by_trip_id_batch");

  return motis_call(make_msg(mc))->val();
}

Offset<Train> trains_response_builder::write_railviz_train(
    message_creator& mc, trip const* trp, size_t const section_index,
    flatbuffers::Vector<int64_t> const* polyline_indices) {
  utl::verify(
      std::distance(sections::begin(trp), sections::end(trp)) > section_index,
      "trains_response_builder: invakud section idx");
  auto const s = std::next(sections::begin(trp), section_index);
  auto const dep = (*s).ev_key_from();
  auto const arr = (*s).ev_key_to();

  auto const dep_di = get_delay_info(sched_, dep);
  auto const arr_di = get_delay_info(sched_, arr);

  auto const dep_station_idx =
      station_indices_.emplace_back(dep.get_station_idx());
  auto const arr_station_idx =
      station_indices_.emplace_back(arr.get_station_idx());

  std::vector<Offset<String>> service_names;
  auto c_info = dep.lcon()->full_con_->con_info_;
  while (c_info != nullptr) {
    service_names.push_back(mc.CreateString(get_service_name(sched_, c_info)));
    c_info = c_info->merged_with_;
  }

  auto const trips =
      utl::to_vec(*sched_.merged_trips_[dep.lcon()->trips_],
                  [&](trip const* trp) { return to_fbs(sched_, mc, trp); });

  return CreateTrain(
      mc, mc.CreateVector(service_names), dep.lcon()->full_con_->clasz_,
      mc.CreateString(sched_.stations_.at(dep_station_idx)->eva_nr_),
      mc.CreateString(sched_.stations_.at(arr_station_idx)->eva_nr_),
      motis_to_unixtime(sched_, dep.get_time()),
      motis_to_unixtime(sched_, arr.get_time()),
      motis_to_unixtime(sched_, dep_di.get_schedule_time()),
      motis_to_unixtime(sched_, arr_di.get_schedule_time()),
      to_fbs(dep_di.get_reason()), to_fbs(arr_di.get_reason()),
      mc.CreateVector(trips),
      mc.CreateVector<int64_t>(polyline_indices->data(),
                               polyline_indices->size()));
}

msg_ptr trains_response_builder::finish() {
  auto const resolved_paths = resolve_paths();
  using motis::path::PathByTripIdBatchResponse;
  auto const* resp = motis_content(PathByTripIdBatchResponse, resolved_paths);

  message_creator mc;

  std::vector<Offset<Train>> trains;
  trains.reserve(queries_.size());

  utl::verify(queries_.size() == resp->segments()->size(),
              "trains_response_builder: path response size mismatch");
  for (auto i = 0ULL; i < queries_.size(); ++i) {
    auto const& query = queries_[i];
    auto const* segment = resp->segments()->Get(i);
    trains.emplace_back(
        write_railviz_train(mc, query.first, query.second, segment->indices()));
  }

  utl::erase_duplicates(station_indices_);
  auto const stations = utl::to_vec(station_indices_, [&](auto s) {
    return to_fbs(mc, *sched_.stations_.at(s));
  });

  auto const polylines = utl::to_vec(*resp->polylines(), [&](auto const* p) {
    return mc.CreateString(p->data(), p->size());
  });

  mc.create_and_finish(
      MsgContent_RailVizTrainsResponse,
      CreateRailVizTrainsResponse(
          mc, mc.CreateVector(stations), mc.CreateVector(trains),
          mc.CreateVector(polylines),
          mc.CreateVector(resp->extras()->data(), resp->extras()->size()))
          .Union());
  return make_msg(mc);
}

}  // namespace motis::railviz
