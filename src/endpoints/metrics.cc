#include "motis/endpoints/metrics.h"

#include <chrono>
#include <functional>
#include <iostream>

#include "prometheus/registry.h"
#include "prometheus/text_serializer.h"

#include "utl/enumerate.h"

#include "nigiri/rt/frun.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"
#include "nigiri/timetable_metrics.h"
#include "nigiri/types.h"

#include "motis/data.h"
#include "motis/tag_lookup.h"

namespace n = nigiri;

namespace motis::ep {

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

  auto metric_by_agency =
      std::vector<std::pair<std::reference_wrapper<prometheus::Gauge>,
                            std::reference_wrapper<prometheus::Gauge>>>{};
  metric_by_agency.reserve(tt.n_agencies());
  for (auto i = nigiri::provider_idx_t{0}; i < tt.n_agencies(); ++i) {
    auto const& p = tt.providers_[i];
    auto const agency_name = tt.strings_.get(p.long_name_);
    auto const agency_id = tt.strings_.get(p.short_name_);
    auto const labels =
        prometheus::Labels{{"tag", std::string{tags.get_tag(p.src_)}},
                           {"agency_name", std::string{agency_name}},
                           {"agency_id", std::string{agency_id}}};
    auto& sched = metrics.current_trips_running_scheduled_count_.Add(labels);
    auto& real =
        metrics.current_trips_running_scheduled_with_realtime_count_.Add(
            labels);
    sched.Set(0);
    real.Set(0);
    metric_by_agency.emplace_back(std::ref(sched), std::ref(real));
  }

  if (rtt != nullptr) {
    for (auto rt_t = nigiri::rt_transport_idx_t{0};
         rt_t < rtt->n_rt_transports(); ++rt_t) {
      auto const fr = n::rt::frun::from_rt(tt, rtt, rt_t);
      if (!fr.is_scheduled()) {
        continue;
      }
      auto const active = n::interval{
          fr[0].time(n::event_type::kDep),
          fr[static_cast<n::stop_idx_t>(fr.stop_range_.size() - 1)].time(
              n::event_type::kArr) +
              n::unixtime_t::duration{1}};
      if (active.overlaps(time_interval)) {
        auto const provider_idx = fr[0].get_provider_idx();
        if (provider_idx != n::provider_idx_t::invalid()) {
          metric_by_agency.at(provider_idx.v_).first.get().Increment();
          metric_by_agency.at(provider_idx.v_).second.get().Increment();
        }
      }
    }
  }

  auto const [start_day, _] = tt.day_idx_mam(time_interval.from_);
  auto const [end_day, _1] = tt.day_idx_mam(time_interval.to_);

  for (auto r = nigiri::route_idx_t{0}; r < tt.n_routes(); ++r) {
    auto const is_active = [&](n::transport const t) -> bool {
      return (rtt == nullptr
                  ? tt.bitfields_[tt.transport_traffic_days_[t.t_idx_]]
                  : rtt->bitfields_[rtt->transport_traffic_days_[t.t_idx_]])
          .test(to_idx(t.day_));
    };

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
        if (is_active(t) &&
            time_interval.overlaps({tt.event_time(t, from, n::event_type::kDep),
                                    tt.event_time(t, to, n::event_type::kArr) +
                                        n::unixtime_t::duration{1}})) {
          auto fr = n::rt::frun::from_t(tt, nullptr, t);
          auto const provider_idx = fr[0].get_provider_idx();
          if (provider_idx != n::provider_idx_t::invalid()) {
            metric_by_agency.at(provider_idx.v_).first.get().Increment();
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
  utl::verify(metrics_ != nullptr && tt_ != nullptr && tags_ != nullptr,
              "no metrics initialized");
  auto const rt = rt_;
  update_all_runs_metrics(*tt_, rt->rtt_.get(), *tags_, *metrics_);
  metrics_->total_trips_with_realtime_count_.Set(
      static_cast<double>(rt->rtt_->rt_transport_src_.size()));
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
