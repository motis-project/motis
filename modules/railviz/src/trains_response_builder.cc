#include "motis/railviz/trains_response_builder.h"

#include <numeric>

#include "geo/polyline_format.h"

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

#include "motis/path/path_database_query.h"

using namespace motis::access;
using namespace motis::logging;
using namespace motis::module;
using namespace flatbuffers;

namespace motis::railviz {

void trains_response_builder::add_train_full(ev_key k) {
  for (auto const& e : route_bfs(k, bfs_direction::BOTH, false)) {
    add_train(train{ev_key{e, k.lcon_idx_, event_type::DEP}, 0.});
  }
}

void trains_response_builder::add_train(train t) {
  auto const& trp = sched_.merged_trips_.at(t.key_.lcon()->trips_)->at(0);

  auto const it = std::find_if(
      sections::begin(trp), sections::end(trp),
      [&](auto const& s) { return s.edge_ == t.key_.route_edge_.get_edge(); });
  utl::verify(it != sections::end(trp),
              "trains_response_builder: missing edge");

  queries_.emplace_back(query{trp, std::distance(sections::begin(trp), it), t});
}

void trains_response_builder::resolve_paths() {
  utl::erase_duplicates(queries_);  // this sorts!

  if (path_data_ == nullptr) {
    return resolve_paths_fallback();
  }

  path::path_database_query path_query{zoom_level_};
  utl::equal_ranges_linear(
      queries_,
      [](auto const& lhs, auto const& rhs) { return lhs.trp_ == rhs.trp_; },
      [&](auto lb, auto ub) {
        try {
          path_query.add_sequence(
              path_data_->trip_to_index(sched_, lb->trp_),
              utl::to_vec(lb, ub, [](auto const& q) -> size_t {
                return q.section_index_;
              }));
        } catch (std::system_error const&) {
          std::vector<geo::polyline> extra;
          for (auto it = lb; it != ub; ++it) {
            auto const& sec =
                *std::next(sections::begin(it->trp_), it->section_index_);
            auto const& s_dep = sec.from_station(sched_);
            auto const& s_arr = sec.to_station(sched_);

            geo::polyline stub;
            stub.emplace_back(s_dep.lat(), s_dep.lng());
            stub.emplace_back(s_arr.lat(), s_arr.lng());
            extra.emplace_back(std::move(stub));
          }
          path_query.add_extra(extra);
        }
      });
  path_query.execute(*path_data_->db_);
  path_query.write_batch(mc_, fbs_segments_, fbs_polylines_, fbs_extras_);
}

void trains_response_builder::resolve_paths_fallback() {
  geo::polyline_encoder<6> enc;
  fbs_polylines_.emplace_back(mc_.CreateString(std::string{}));  // no zero

  fbs_segments_ = utl::to_vec(queries_, [&](auto const& q) {
    utl::verify(std::distance(sections::begin(q.trp_), sections::end(q.trp_)) >
                    q.section_index_,
                "trains_response_builder: invakud section idx");
    auto const s = std::next(sections::begin(q.trp_), q.section_index_);
    auto const s_min = std::min((*s).from_station_id(), (*s).to_station_id());
    auto const s_max = std::max((*s).from_station_id(), (*s).to_station_id());

    std::vector<int64_t> indices;
    indices.push_back(static_cast<int64_t>(utl::get_or_create(
                          fallback_indices_, std::make_pair(s_min, s_max),
                          [&] {
                            enc.push({sched_.stations_[s_min]->lat(),
                                      sched_.stations_[s_min]->lng()});
                            enc.push({sched_.stations_[s_max]->lat(),
                                      sched_.stations_[s_max]->lng()});
                            fbs_polylines_.emplace_back(
                                mc_.CreateString(enc.buf_));
                            enc.reset();
                            return fbs_polylines_.size() - 1;
                          })) *
                      ((s_min != (*s).from_station_id()) ? -1 : 1));
    return indices;
  });

  fbs_extras_.resize(fbs_polylines_.size() - 1);
  std::iota(begin(fbs_extras_), end(fbs_extras_), 1);
}

Offset<Train> trains_response_builder::write_railviz_train(
    query const& q, std::vector<int64_t> const& polyline_indices) {
  utl::verify(std::distance(sections::begin(q.trp_), sections::end(q.trp_)) >
                  q.section_index_,
              "trains_response_builder: invakud section idx");
  auto const s = std::next(sections::begin(q.trp_), q.section_index_);
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
    service_names.push_back(mc_.CreateString(get_service_name(sched_, c_info)));
    c_info = c_info->merged_with_;
  }

  auto const trips =
      utl::to_vec(*sched_.merged_trips_[dep.lcon()->trips_],
                  [&](trip const* trp) { return to_fbs(sched_, mc_, trp); });

  return CreateTrain(
      mc_, mc_.CreateVector(service_names),
      static_cast<service_class_t>(dep.lcon()->full_con_->clasz_),
      q.train_.route_distance_,
      mc_.CreateString(sched_.stations_.at(dep_station_idx)->eva_nr_),
      mc_.CreateString(sched_.stations_.at(arr_station_idx)->eva_nr_),
      motis_to_unixtime(sched_, dep.get_time()),
      motis_to_unixtime(sched_, arr.get_time()),
      motis_to_unixtime(sched_, dep_di.get_schedule_time()),
      motis_to_unixtime(sched_, arr_di.get_schedule_time()),
      to_fbs(dep_di.get_reason()), to_fbs(arr_di.get_reason()),
      mc_.CreateVector(trips), mc_.CreateVector(polyline_indices));
}

msg_ptr trains_response_builder::finish() {
  resolve_paths();

  std::vector<Offset<Train>> trains;
  trains.reserve(queries_.size());

  utl::verify(queries_.size() == fbs_segments_.size(),
              "trains_response_builder: query segment size mismatch mismatch");
  for (auto i = 0ULL; i < queries_.size(); ++i) {
    trains.emplace_back(write_railviz_train(queries_[i], fbs_segments_[i]));
  }

  utl::erase_duplicates(station_indices_);
  auto const stations = utl::to_vec(station_indices_, [&](auto s) {
    return to_fbs(mc_, *sched_.stations_.at(s));
  });

  mc_.create_and_finish(
      MsgContent_RailVizTrainsResponse,
      CreateRailVizTrainsResponse(
          mc_, mc_.CreateVector(stations), mc_.CreateVector(trains),
          mc_.CreateVector(fbs_polylines_), mc_.CreateVector(fbs_extras_))
          .Union());
  return make_msg(mc_);
}

}  // namespace motis::railviz
