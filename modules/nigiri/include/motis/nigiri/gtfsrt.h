#pragma once

#include <memory>
#include <string>
#include <string_view>

#include "prometheus/counter.h"

#include "motis/module/context/motis_http_req.h"
#include "motis/nigiri/tag_lookup.h"

namespace nigiri::rt {
struct statistics;
}

namespace motis::nigiri {

struct metrics;

struct gtfsrt_metrics {
  prometheus::Counter& updates_requested_;
  prometheus::Counter& updates_successful_;
  prometheus::Counter& updates_error_;

  prometheus::Counter& total_entities_;
  prometheus::Counter& total_entities_success_;
  prometheus::Counter& total_entities_fail_;
  prometheus::Counter& unsupported_deleted_;
  prometheus::Counter& unsupported_vehicle_;
  prometheus::Counter& unsupported_alert_;
  prometheus::Counter& unsupported_no_trip_id_;
  prometheus::Counter& no_trip_update_;
  prometheus::Counter& trip_update_without_trip_;
  prometheus::Counter& trip_resolve_error_;
  prometheus::Counter& unsupported_schedule_relationship_;
  prometheus::Gauge& feed_timestamp_;
  prometheus::Gauge& last_update_timestamp_;
};

struct gtfsrt {
  // Config format: tag|url|auth
  // Example 1: nl|http://gtfs.ovapi.nl/nl/tripUpdates.pb|my_api_key
  // Example 2: nl|http://gtfs.ovapi.nl/nl/tripUpdates.pb
  gtfsrt(tag_lookup const&, std::string_view config, metrics&);
  gtfsrt(gtfsrt&&) noexcept;
  gtfsrt& operator=(gtfsrt&&) noexcept;
  ~gtfsrt();

  motis::module::http_future_t fetch() const;
  ::nigiri::source_idx_t src() const;

  void update_metrics(::nigiri::rt::statistics const&) const;

  struct impl;
  std::unique_ptr<impl> impl_;

  std::string tag_;
  std::string url_;
  std::unique_ptr<gtfsrt_metrics> metrics_;
};

}  // namespace motis::nigiri
