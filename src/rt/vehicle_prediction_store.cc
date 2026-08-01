#include "motis/rt/vehicle_prediction_store.h"

#include <algorithm>
#include <tuple>
#include <utility>

namespace motis {
namespace {

auto key(vehicle_prediction_diagnostic_entry const& x) {
  return std::tuple{x.transport_, x.static_stop_sequence_};
}

}  // namespace

std::unique_ptr<vehicle_prediction_diagnostics_store>
vehicle_prediction_diagnostics_store::build(
    bool const enabled,
    std::vector<vehicle_prediction_diagnostic_entry> entries,
    std::int64_t const now_seconds) {
  return build(enabled, std::move(entries), now_seconds, limits{});
}

std::unique_ptr<vehicle_prediction_diagnostics_store>
vehicle_prediction_diagnostics_store::build(
    bool const enabled,
    std::vector<vehicle_prediction_diagnostic_entry> entries,
    std::int64_t const now_seconds,
    limits const policy) {
  if (!enabled) {
    return nullptr;
  }
  std::erase_if(entries, [&](auto const& x) {
    return x.transport_ == nigiri::transport::invalid() ||
           x.trip_id_.empty() || now_seconds < x.observed_at_seconds_ ||
           now_seconds - x.observed_at_seconds_ > policy.max_age_seconds_;
  });
  std::ranges::sort(entries, [](auto const& a, auto const& b) {
    if (key(a) != key(b)) {
      return key(a) < key(b);
    }
    return a.observed_at_seconds_ > b.observed_at_seconds_;
  });
  entries.erase(std::unique(begin(entries), end(entries), [](auto const& a,
                                                             auto const& b) {
                  return key(a) == key(b);
                }),
                end(entries));
  if (entries.size() > policy.max_entries_) {
    entries.resize(policy.max_entries_);
  }
  auto store = std::make_unique<vehicle_prediction_diagnostics_store>();
  store->entries_ = std::move(entries);
  return store;
}

vehicle_prediction_diagnostic_entry const*
vehicle_prediction_diagnostics_store::find(
    nigiri::transport const transport,
    unsigned const static_stop_sequence) const {
  auto const lookup = std::tuple{transport, static_stop_sequence};
  auto const it = std::ranges::lower_bound(
      entries_, lookup, {}, [](auto const& x) { return key(x); });
  return it != end(entries_) && key(*it) == lookup ? &*it : nullptr;
}

vehicle_prediction_diagnostic_entry const*
vehicle_prediction_diagnostics_store::find_event(
    nigiri::transport const transport,
    std::string_view const trip_id,
    std::int64_t const scheduled_timestamp_seconds) const {
  auto it = std::ranges::lower_bound(
      entries_, transport, {},
      &vehicle_prediction_diagnostic_entry::transport_);
  for (; it != end(entries_) && it->transport_ == transport; ++it) {
    auto const scheduled = it->effective_.predicted_timestamp_seconds_ -
                           it->effective_.delay_seconds_;
    if (it->trip_id_ == trip_id && scheduled == scheduled_timestamp_seconds) {
      return &*it;
    }
  }
  return nullptr;
}

}  // namespace motis
