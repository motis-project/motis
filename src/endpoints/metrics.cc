#include "motis/endpoints/metrics.h"

#include <array>
#include <chrono>
#include <iostream>
#include <mutex>
#include <optional>

#include "fmt/ostream.h"

#include "prometheus/registry.h"
#include "prometheus/text_serializer.h"

#include "utl/enumerate.h"

#include "nigiri/rt/frun.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"
#include "nigiri/timetable_metrics.h"
#include "nigiri/types.h"

#include "motis/data.h"
#include "motis/place.h"
#include "motis/tag_lookup.h"
#include "motis/timetable/clasz_to_mode.h"

namespace n = nigiri;

namespace motis::ep {

// Whole minutes, le=N counts delays <= N.
constexpr auto const kDelayBuckets = std::array<double, 18U>{
    -10, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 60, 90};

std::optional<std::pair<n::rt::run_stop, n::event_type>> current_event(
    n::rt::frun const& fr, n::unixtime_t const now) {
  auto event = std::optional<std::pair<n::rt::run_stop, n::event_type>>{};
  for (auto const stp : fr) {
    if (stp.is_cancelled()) {
      continue;
    }
    for (auto const ev : {n::event_type::kArr, n::event_type::kDep}) {
      if ((ev == n::event_type::kArr &&
           stp.stop_idx_ == fr.stop_range_.from_) ||
          (ev == n::event_type::kDep &&
           stp.stop_idx_ + 1U == fr.stop_range_.to_)) {
        continue;
      }
      event = {stp, ev};
      if (stp.time(ev) >= now) {
        return event;
      }
    }
  }
  return event;
}

void update_all_runs_metrics(nigiri::timetable const& tt,
                             nigiri::rt_timetable const* rtt,
                             tag_lookup const& tags,
                             metrics_registry& metrics) {
  auto const start_time =
      std::max(std::chrono::time_point_cast<n::unixtime_t::duration>(
                   std::chrono::system_clock::now()),
               tt.external_interval().from_);
  auto const end_time = std::min(
      std::chrono::time_point_cast<n::unixtime_t::duration>(
          start_time +
          std::chrono::duration_cast<n::duration_t>(std::chrono::minutes{3})),
      tt.external_interval().to_);
  auto const time_interval = n::interval{start_time, end_time};

  constexpr auto const kRtDataStates =
      std::array{n::rt_data_state::kNoRtData, n::rt_data_state::kInconsistent,
                 n::rt_data_state::kObserved, n::rt_data_state::kPropagated,
                 n::rt_data_state::kPredicted};

  for (auto& [_, m] : metrics.running_trips_) {
    for (auto* g : {&m.scheduled_, &m.realtime_, &m.cancelled_, &m.added_}) {
      g->Set(0);
    }
    for (auto* g : m.rt_events_) {
      g->Set(0);
    }
    if (m.delay_ != nullptr) {
      m.delay_->Reset();
    }
  }

  auto const get = [&](n::rt::run_stop const& s) -> running_trips_metrics* {
    auto const provider_idx = s.get_provider_idx(n::event_type::kDep);
    if (provider_idx == n::provider_idx_t::invalid()) {
      return nullptr;
    }
    auto const clasz = s.get_clasz(n::event_type::kDep);
    auto const key =
        std::pair{to_idx(provider_idx), static_cast<std::uint8_t>(clasz)};
    if (auto const it = metrics.running_trips_.find(key);
        it != end(metrics.running_trips_)) {
      return &it->second;
    }

    auto const& p = tt.providers_[provider_idx];
    auto const labels = prometheus::Labels{
        {"tag", std::string{tags.get_tag(p.src_)}},
        {"agency_name", std::string{tt.get_default_translation(p.name_)}},
        {"agency_id", std::string{tt.strings_.get(p.id_)}},
        {"mode", fmt::to_string(fmt::streamed(to_mode(clasz, 5U)))}};
    auto const with = [&](std::string const& k, std::string const& v) {
      auto l = labels;
      l.emplace(k, v);
      return l;
    };

    auto rt_events = std::vector<prometheus::Gauge*>{};
    for (auto const state : kRtDataStates) {
      rt_events.push_back(&metrics.total_rt_events_count_.Add(with(
          "timestamp_reason", fmt::to_string(fmt::streamed(to_api(state))))));
    }

    auto& rt = metrics.current_trips_running_scheduled_with_realtime_count_;
    auto const [it, _] = metrics.running_trips_.emplace(
        key,
        running_trips_metrics{
            labels, metrics.current_trips_running_scheduled_count_.Add(labels),
            rt.Add(with("cancelled", "false")),
            rt.Add(with("cancelled", "true")),
            metrics.current_trips_running_added_count_.Add(labels),
            std::move(rt_events)});
    return &it->second;
  };

  if (rtt != nullptr) {
    for (auto rt_t = nigiri::rt_transport_idx_t{0};
         rt_t < rtt->n_rt_transports(); ++rt_t) {
      auto const fr = n::rt::frun::from_rt(tt, rtt, rt_t);
      auto* const m = get(fr[0]);
      if (m == nullptr) {
        continue;
      }
      for (auto const state : rtt->rt_transport_data_states_[rt_t]) {
        m->rt_events_[static_cast<std::size_t>(state)]->Increment();
      }
      auto const active = n::interval{
          fr[0].time(n::event_type::kDep),
          fr[static_cast<n::stop_idx_t>(fr.stop_range_.size() - 1)].time(
              n::event_type::kArr) +
              n::unixtime_t::duration{1}};
      if (!active.overlaps(time_interval)) {
        continue;
      }
      if (!fr.is_scheduled()) {
        if (!fr.is_cancelled()) {
          m->added_.Increment();
        }
        continue;
      }
      m->scheduled_.Increment();
      if (fr.is_cancelled()) {
        m->cancelled_.Increment();
        continue;
      }
      m->realtime_.Increment();
      if (auto const e = current_event(fr, start_time); e.has_value()) {
        auto const& [stp, ev] = *e;
        auto const delay = static_cast<double>(stp.delay(ev).count());
        if (m->delay_ == nullptr) {
          m->delay_ = &metrics.current_trips_running_delay_minutes_.Add(
              m->labels_, prometheus::Histogram::BucketBoundaries{
                              begin(kDelayBuckets), end(kDelayBuckets)});
        }
        m->delay_->Observe(delay);
      }
    }
  }

  auto const [start_day, _] = tt.day_idx_mam(time_interval.from_);
  auto const [end_day, _1] = tt.day_idx_mam(time_interval.to_);

  for (auto r = nigiri::route_idx_t{0}; r < tt.n_routes(); ++r) {
    auto const seq = tt.route_location_seq_[r];
    auto const from = n::stop_idx_t{0U};
    auto const to = static_cast<n::stop_idx_t>(seq.size() - 1);
    auto const arr_times = tt.event_times_at_stop(r, to, n::event_type::kArr);

    for (auto const [i, t_idx] :
         utl::enumerate(tt.route_transport_ranges_[r])) {
      auto const day_offset =
          static_cast<n::day_idx_t::value_t>(arr_times[i].days());
      for (auto day = start_day - day_offset; day <= end_day; ++day) {
        auto const t = n::transport{t_idx, day};
        if (tt.is_transport_active(t.t_idx_, t.day_) &&
            (rtt == nullptr ||
             rtt->resolve_rt(t) == n::rt_transport_idx_t::invalid()) &&
            time_interval.overlaps({tt.event_time(t, from, n::event_type::kDep),
                                    tt.event_time(t, to, n::event_type::kArr) +
                                        n::unixtime_t::duration{1}})) {
          auto fr = n::rt::frun::from_t(tt, nullptr, t);
          if (auto* const m = get(fr[0]); m != nullptr) {
            m->scheduled_.Increment();
            if (rtt != nullptr && !rtt->is_transport_active(t.t_idx_, t.day_)) {
              m->cancelled_.Increment();  // cancelled without RT transport
            }
          }
        }
      }
    }
  }

  if (metrics.timetable_first_day_timestamp_.Collect().empty()) {
    auto const m = get_metrics(tt);
    constexpr auto const kEndOfDay =
        std::chrono::days{1} - std::chrono::seconds{1};
    auto const from = std::chrono::duration_cast<std::chrono::seconds>(
        tt.internal_interval().from_.time_since_epoch());
    for (auto src = n::source_idx_t{0U}; src != tt.n_sources(); ++src) {
      auto const& fm = m.feeds_[src];
      auto const labels =
          prometheus::Labels{{"tag", std::string{tags.get_tag(src)}}};
      metrics.timetable_first_day_timestamp_.Add(
          labels, static_cast<double>((from + date::days{fm.first_}).count()));
      metrics.timetable_last_day_timestamp_.Add(
          labels, static_cast<double>(
                      (from + date::days{fm.last_} + kEndOfDay).count()));
      metrics.timetable_locations_count_.Add(labels, fm.locations_);
      metrics.timetable_trips_count_.Add(labels, fm.trips_);
      metrics.timetable_transports_x_days_count_.Add(
          labels, static_cast<double>(fm.transport_days_));
    }
  }
}

net::reply metrics::operator()(net::route_request const& req, bool) const {
  utl::verify(metrics_ != nullptr, "no metrics initialized");
  auto const lock = std::scoped_lock{metrics_->running_trips_mutex_};
  auto const rt = std::atomic_load(&rt_);
  if (tt_ != nullptr && tags_ != nullptr) {
    update_all_runs_metrics(*tt_, rt->rtt_.get(), *tags_, *metrics_);
  }
  if (rt != nullptr && rt->rtt_ != nullptr) {
    metrics_->total_trips_with_realtime_count_.Set(
        static_cast<double>(rt->rtt_->rt_transport_src_.size()));
  }
  auto res = net::web_server::string_res_t{boost::beast::http::status::ok,
                                           req.version()};
  res.insert(boost::beast::http::field::content_type,
             "text/plain; version=0.0.4");
  set_response_body(
      res, req,
      prometheus::TextSerializer{}.Serialize(metrics_->registry_.Collect()));
  res.keep_alive(req.keep_alive());
  return res;
}

}  // namespace motis::ep
