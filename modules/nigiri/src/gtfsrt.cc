#include "motis/nigiri/gtfsrt.h"

#include "utl/parser/split.h"

#include "net/http/client/request.h"

#include "prometheus/registry.h"

#include "nigiri/rt/gtfsrt_update.h"

#include "motis/nigiri/location.h"
#include "motis/nigiri/metrics.h"

namespace mm = motis::module;
namespace n = nigiri;

namespace motis::nigiri {

struct gtfsrt::impl {
  impl(net::http::client::request req, n::source_idx_t const src)
      : req_{std::move(req)}, src_{src} {}
  net::http::client::request req_;
  n::source_idx_t src_;
};

gtfsrt::gtfsrt(tag_lookup const& tags, std::string_view config,
               metrics& metrics) {
  auto const [tag, url, auth] =
      utl::split<'|', utl::cstr, utl::cstr, utl::cstr>(config);
  tag_ = tag.to_str();
  url_ = url.to_str();
  auto const src = tags.get_src(tag.to_str() + "_");
  utl::verify(
      src != n::source_idx_t::invalid(),
      "nigiri GTFS-RT tag {} not found as static timetable (known tags: {})",
      tag.view(), fmt::streamed(tags));
  auto req = net::http::client::request{url.to_str()};
  if (!auth.empty()) {
    url.starts_with("https://gtfs-datenstroeme.tech.deutschebahn.com")
        ? req.headers.emplace("DB-Api-Key", auth.to_str())
        : req.headers.emplace("Authorization", auth.to_str());
  }
  impl_ = std::make_unique<impl>(std::move(req), src);
  metrics_ = std::make_unique<gtfsrt_metrics>(gtfsrt_metrics{
      .updates_requested_ =
          metrics.gtfsrt_updates_requested_.Add({{"tag", tag.to_str()}}),
      .updates_successful_ =
          metrics.gtfsrt_updates_successful_.Add({{"tag", tag.to_str()}}),
      .updates_error_ =
          metrics.gtfsrt_updates_error_.Add({{"tag", tag.to_str()}}),
      .total_entities_ =
          metrics.gtfsrt_total_entities_.Add({{"tag", tag.to_str()}}),
      .total_entities_success_ =
          metrics.gtfsrt_total_entities_success_.Add({{"tag", tag.to_str()}}),
      .total_entities_fail_ =
          metrics.gtfsrt_total_entities_fail_.Add({{"tag", tag.to_str()}}),
      .unsupported_deleted_ =
          metrics.gtfsrt_unsupported_deleted_.Add({{"tag", tag.to_str()}}),
      .unsupported_vehicle_ =
          metrics.gtfsrt_unsupported_vehicle_.Add({{"tag", tag.to_str()}}),
      .unsupported_alert_ =
          metrics.gtfsrt_unsupported_alert_.Add({{"tag", tag.to_str()}}),
      .unsupported_no_trip_id_ =
          metrics.gtfsrt_unsupported_no_trip_id_.Add({{"tag", tag.to_str()}}),
      .no_trip_update_ =
          metrics.gtfsrt_no_trip_update_.Add({{"tag", tag.to_str()}}),
      .trip_update_without_trip_ =
          metrics.gtfsrt_trip_update_without_trip_.Add({{"tag", tag.to_str()}}),
      .trip_resolve_error_ =
          metrics.gtfsrt_trip_resolve_error_.Add({{"tag", tag.to_str()}}),
      .unsupported_schedule_relationship_ =
          metrics.gtfsrt_unsupported_schedule_relationship_.Add(
              {{"tag", tag.to_str()}}),
      .feed_timestamp_ =
          metrics.gtfsrt_feed_timestamp_.Add({{"tag", tag.to_str()}}),
      .last_update_timestamp_ =
          metrics.gtfsrt_last_update_timestamp_.Add({{"tag", tag.to_str()}}),
  });
}

gtfsrt::gtfsrt(gtfsrt&&) noexcept = default;
gtfsrt& gtfsrt::operator=(gtfsrt&&) noexcept = default;

gtfsrt::~gtfsrt() = default;

mm::http_future_t gtfsrt::fetch() const { return motis_http(impl_->req_); }

n::source_idx_t gtfsrt::src() const { return impl_->src_; }

void gtfsrt::update_metrics(::nigiri::rt::statistics const& stats) const {
  metrics_->total_entities_.Increment(stats.total_entities_);
  metrics_->total_entities_success_.Increment(stats.total_entities_success_);
  metrics_->total_entities_fail_.Increment(stats.total_entities_fail_);
  metrics_->unsupported_deleted_.Increment(stats.unsupported_deleted_);
  metrics_->unsupported_vehicle_.Increment(stats.unsupported_vehicle_);
  metrics_->unsupported_alert_.Increment(stats.unsupported_alert_);
  metrics_->unsupported_no_trip_id_.Increment(stats.unsupported_no_trip_id_);
  metrics_->no_trip_update_.Increment(stats.no_trip_update_);
  metrics_->trip_update_without_trip_.Increment(
      stats.trip_update_without_trip_);
  metrics_->trip_resolve_error_.Increment(stats.trip_resolve_error_);
  metrics_->unsupported_schedule_relationship_.Increment(
      stats.unsupported_schedule_relationship_);
  metrics_->feed_timestamp_.Set(
      static_cast<double>(stats.feed_timestamp_.time_since_epoch().count()));
}

}  // namespace motis::nigiri
