#include "motis/endpoints/gtfsrt.h"

#include <functional>

#ifdef NO_DATA
#undef NO_DATA
#endif
#include "gtfsrt/gtfs-realtime.pb.h"

#include "utl/enumerate.h"

#include "net/too_many_exception.h"

#include "nigiri/loader/gtfs/stop_seq_number_encoding.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

#include "motis/data.h"
#include "motis/tag_lookup.h"

namespace n = nigiri;

namespace gtfsrt = transit_realtime;
namespace protob = google::protobuf;

namespace motis::ep {

void add_trip_updates(n::timetable const& tt,
                      tag_lookup const& tags,
                      nigiri::rt::frun const& fr,
                      transit_realtime::FeedMessage& fm) {
  fr.for_each_trip([&](n::trip_idx_t trip_idx,
                       n::interval<n::stop_idx_t> const subrange) {
    auto fe = fm.add_entity();
    auto const trip_id = tags.id_fragments(
        tt, fr[subrange.from_ - fr.stop_range_.from_], n::event_type::kDep);
    fe->set_id(trip_id.trip_id_);
    auto tu = fe->mutable_trip_update();
    auto td = tu->mutable_trip();
    td->set_trip_id(trip_id.trip_id_);
    td->set_start_time(fmt::format("{}:00", trip_id.start_time_));
    td->set_start_date(trip_id.start_date_);
    td->set_schedule_relationship(
        fr.is_cancelled()
            ? transit_realtime::TripDescriptor_ScheduleRelationship::
                  TripDescriptor_ScheduleRelationship_CANCELED
        : !fr.is_scheduled()
            ? transit_realtime::TripDescriptor_ScheduleRelationship::
                  TripDescriptor_ScheduleRelationship_ADDED
            : transit_realtime::TripDescriptor_ScheduleRelationship::
                  TripDescriptor_ScheduleRelationship_SCHEDULED);
    if (!fr.is_scheduled()) {
      auto const route_id_idx = fr.rtt_->rt_transport_route_id_.at(fr.rt_);
      if (route_id_idx != n::route_id_idx_t::invalid()) {
        td->set_route_id(
            tt.route_ids_[fr.rtt_->rt_transport_src_.at(fr.rt_)].ids_.get(
                route_id_idx));
      }
    }
    if (fr.is_cancelled()) {
      return;
    }
    auto const seq_numbers =
        fr.is_scheduled()
            ? n::loader::gtfs::
                  stop_seq_number_range{{tt.trip_stop_seq_numbers_[trip_idx]},
                                        static_cast<n::stop_idx_t>(
                                            subrange.size())}
            : n::loader::gtfs::stop_seq_number_range{
                  std::span<n::stop_idx_t>{},
                  static_cast<n::stop_idx_t>(fr.size())};
    auto stop_idx =
        fr.is_scheduled() ? subrange.from_ : static_cast<unsigned short>(0U);
    auto seq_it = begin(seq_numbers);

    auto last_delay = n::duration_t::max();
    for (; seq_it != end(seq_numbers); ++stop_idx, ++seq_it) {
      auto const s = fr[stop_idx - fr.stop_range_.from_];

      transit_realtime::TripUpdate_StopTimeUpdate* stu = nullptr;
      auto const set_stu = [&]() {
        if (stu != nullptr) {
          return;
        }
        stu = tu->add_stop_time_update();
        stu->set_stop_id(
            tt.locations_.ids_[s.get_stop().location_idx()].view());
        stu->set_stop_sequence(*seq_it);
      };

      auto const to_unix = [&](n::unixtime_t t) {
        return std::chrono::time_point_cast<std::chrono::seconds>(t)
            .time_since_epoch()
            .count();
      };
      auto const to_delay_seconds = [&](n::duration_t t) {
        return static_cast<int32_t>(
            std::chrono::duration_cast<std::chrono::seconds>(t).count());
      };

      if (s.stop_idx_ != 0) {
        auto const arr_delay = s.delay(nigiri::event_type::kArr);
        if (arr_delay != last_delay || !fr.is_scheduled()) {
          set_stu();
          auto ar = stu->mutable_arrival();
          ar->set_time(to_unix(s.time(nigiri::event_type::kArr)));
          ar->set_delay(to_delay_seconds(arr_delay));
          last_delay = arr_delay;
        }
      }
      if (s.stop_idx_ != fr.size() - 1) {
        auto const dep_delay = s.delay(nigiri::event_type::kDep);
        if (dep_delay != last_delay || !fr.is_scheduled()) {
          set_stu();
          auto dep = stu->mutable_departure();
          dep->set_time(to_unix(s.time(nigiri::event_type::kDep)));
          dep->set_delay(to_delay_seconds(dep_delay));
          last_delay = dep_delay;
        }
      }
      if (s.is_cancelled() && !s.get_scheduled_stop().is_cancelled()) {
        set_stu();
        stu->set_schedule_relationship(
            transit_realtime::TripUpdate_StopTimeUpdate_ScheduleRelationship::
                TripUpdate_StopTimeUpdate_ScheduleRelationship_SKIPPED);
      }
    }
  });
}

void add_rt_transports(n::timetable const& tt,
                       tag_lookup const& tags,
                       n::rt_timetable const& rtt,
                       transit_realtime::FeedMessage& fm) {
  for (auto rt_t = nigiri::rt_transport_idx_t{0}; rt_t < rtt.n_rt_transports();
       ++rt_t) {
    auto const fr = n::rt::frun::from_rt(tt, &rtt, rt_t);
    add_trip_updates(tt, tags, fr, fm);
  }
}

void add_cancelled_transports(n::timetable const& tt,
                              tag_lookup const& tags,
                              n::rt_timetable const& rtt,
                              transit_realtime::FeedMessage& fm) {
  auto const start_time = std::max(
      std::chrono::time_point_cast<n::unixtime_t::duration>(
          std::chrono::system_clock::now() -
          std::chrono::duration_cast<n::duration_t>(n::kTimetableOffset)),
      tt.internal_interval().from_);
  auto const end_time = std::min(
      std::chrono::time_point_cast<n::unixtime_t::duration>(
          start_time +
          std::chrono::duration_cast<n::duration_t>(std::chrono::days{6})),
      tt.internal_interval().to_);
  auto const [start_day, _] = tt.day_idx_mam(start_time);
  auto const [end_day, _1] = tt.day_idx_mam(end_time);

  for (auto r = nigiri::route_idx_t{0}; r < tt.n_routes(); ++r) {
    for (auto const [i, t_idx] :
         utl::enumerate(tt.route_transport_ranges_[r])) {
      for (auto day = start_day; day <= end_day; ++day) {
        auto const t = n::transport{t_idx, day};
        auto const is_cancelled =
            tt.bitfields_[tt.transport_traffic_days_[t.t_idx_]].test(
                to_idx(t.day_)) &&
            !rtt.bitfields_[rtt.transport_traffic_days_[t.t_idx_]].test(
                to_idx(t.day_));
        if (!is_cancelled) {
          continue;
        }
        auto fr = n::rt::frun::from_t(tt, &rtt, t);
        if (fr.is_rt()) {
          continue;
        }
        add_trip_updates(tt, tags, fr, fm);
      }
    }
  }
}

net::reply gtfsrt::operator()(net::route_request const& req, bool) const {
  utl::verify(tt_ != nullptr && tags_ != nullptr, "no tt initialized");
  auto const rt = std::atomic_load(&rt_);
  auto const rtt = rt->rtt_.get();

  utl::verify<net::too_many_exception>(
      config_.get_limits().gtfsrt_expose_max_trip_updates_ != 0 &&
          rtt->n_rt_transports() <
              config_.get_limits().gtfsrt_expose_max_trip_updates_,
      "number of trip updates above configured limit");

  auto fm = transit_realtime::FeedMessage();
  auto fh = fm.mutable_header();
  fh->set_gtfs_realtime_version("2.0");
  fh->set_incrementality(
      transit_realtime::FeedHeader_Incrementality_FULL_DATASET);
  auto const time = std::time(nullptr);
  fh->set_timestamp(static_cast<double>(time));

  if (rtt != nullptr) {
    add_rt_transports(*tt_, *tags_, *rtt, fm);
    add_cancelled_transports(*tt_, *tags_, *rtt, fm);
  }

  auto res = net::web_server::string_res_t{boost::beast::http::status::ok,
                                           req.version()};
  res.insert(boost::beast::http::field::content_type, "application/x-protobuf");
  res.keep_alive(req.keep_alive());
  set_response_body(res, req, fm.SerializeAsString());

  return res;
}

}  // namespace motis::ep