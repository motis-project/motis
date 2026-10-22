#include "motis/flex/flex_output.h"

#include <chrono>

#include "date/tz.h"

#include "nigiri/flex.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

#include "motis/flex/flex.h"
#include "motis/flex/flex_areas.h"
#include "motis/flex/flex_routing_data.h"
#include "motis/osr/street_routing.h"
#include "motis/place.h"
#include "motis/to_alert.h"

namespace n = nigiri;

namespace motis::flex {

n::unixtime_t local_time_to_unixtime(date::time_zone const* tz,
                                     date::local_days const day,
                                     n::duration_t const time) {
  return std::chrono::time_point_cast<n::unixtime_t::duration>(
      tz->to_sys(day + time, date::choose::earliest));
}

namespace {

struct service_day {
  date::time_zone const* tz_;
  date::local_days day_;
};

// Returns the GTFS service day (in the timezone of the agency) of the flex trip
// whose time window at `stop` contains `start`.
std::optional<service_day> get_service_day(n::timetable const& tt,
                                           n::flex_transport_idx_t const t,
                                           n::stop_idx_t const stop,
                                           n::unixtime_t const start) {
  auto const trip = tt.flex_transport_trip_[t];
  auto const src = tt.trip_id_src_[tt.trip_ids_[trip].front()];
  auto const provider =
      tt.route_ids_[src].route_id_provider_[tt.trip_route_id_[trip]];
  auto const tz_idx = tt.providers_[provider].tz_;
  if (tz_idx == n::timezone_idx_t::invalid()) {
    return std::nullopt;
  }
  auto const tz = n::to_time_zone(tt.timezones_[tz_idx]);
  if (tz == nullptr) {
    return std::nullopt;
  }

  // nigiri stores the time windows of a flex transport relative to its UTC
  // traffic days. Find the day whose window at `stop` contains `start`,
  // preferring traffic days of the transport. Around DST changes, the routing
  // may use a transport whose traffic days do not contain that day.
  auto const& traffic_days = tt.bitfields_[tt.flex_transport_traffic_days_[t]];
  auto const windows = tt.flex_transport_stop_time_windows_[t];
  auto const base = tt.internal_interval().from_;
  auto const start_day = std::chrono::floor<date::days>(start - base).count();
  for (auto const only_traffic_days : {true, false}) {
    for (auto d = start_day; d >= start_day - 2; --d) {
      if (d < 0 || static_cast<std::size_t>(d) >= n::kMaxDays ||
          (only_traffic_days &&
           !traffic_days.test(static_cast<std::size_t>(d)))) {
        continue;
      }
      auto const day = base + date::days{d};
      if (day + windows[stop].from_ <= start &&
          start <= day + windows[stop].to_) {
        // nigiri does not keep the GTFS service day. It is the local date on
        // which the first time window of the trip starts, given that this
        // window starts before 24:00:00 of the service day.
        return service_day{tz, std::chrono::floor<date::days>(
                                   tz->to_local(day + windows.front().from_))};
      }
    }
  }
  return std::nullopt;
}

// The day `n` days before `day`. If `service` is set, only days of this
// service are counted (GTFS `prior_notice_service_id`).
std::optional<date::local_days> days_before(n::timetable const& tt,
                                            date::local_days const day,
                                            unsigned const n,
                                            n::bitfield_idx_t const service) {
  if (service == n::bitfield_idx_t::invalid()) {
    return day - date::days{n};
  }
  auto const& service_days = tt.bitfields_[service];
  auto const first_day =
      date::local_days{tt.internal_interval_days().from_.time_since_epoch()};
  auto d = day;
  for (auto i = 0U; i != n;) {
    d -= date::days{1};
    auto const idx = (d - first_day).count();
    if (idx < 0) {
      return std::nullopt;
    }
    if (static_cast<std::size_t>(idx) < n::kMaxDays &&
        service_days.test(static_cast<std::size_t>(idx))) {
      ++i;
    }
  }
  return d;
}

std::optional<api::BookingRule> get_booking_rule(
    n::timetable const& tt,
    n::lang_t const& lang,
    n::booking_rule_idx_t const idx,
    api::Leg const& leg,
    std::optional<service_day> const& sd) {
  if (idx == n::booking_rule_idx_t::invalid()) {
    return std::nullopt;
  }

  auto const translate =
      [&](n::translation_idx_t const t) -> std::optional<std::string> {
    if (t == n::translation_idx_t::invalid()) {
      return std::nullopt;
    }
    auto const s = tt.translate(lang, t);
    return s.empty() ? std::nullopt : std::optional{std::string{s}};
  };
  auto const days_before_service_day =
      [&](unsigned const n,
          n::bitfield_idx_t const service) -> std::optional<date::local_days> {
    return sd.has_value() ? days_before(tt, sd->day_, n, service)
                          : std::nullopt;
  };

  auto const& r = tt.booking_rules_[idx];
  auto rule = api::BookingRule{};
  r.type_.apply(utl::overloaded{
      [&](n::booking_rule::real_time const&) {
        rule.bookingType_ = api::BookingTypeEnum::REAL_TIME;
      },
      [&](n::booking_rule::prior_notice const& x) {
        rule.bookingType_ = api::BookingTypeEnum::UP_TO_SAME_DAY;
        rule.priorNoticeDurationMin_ = x.prior_notice_duration_min_.count();
        rule.latestBookingTime_ =
            leg.startTime_.time_ - x.prior_notice_duration_min_;
        if (x.prior_notice_duration_max_ !=
            n::booking_rule::prior_notice{}.prior_notice_duration_max_) {
          rule.priorNoticeDurationMax_ = x.prior_notice_duration_max_.count();
          rule.earliestBookingTime_ =
              leg.startTime_.time_ - x.prior_notice_duration_max_;
        }
      },
      [&](n::booking_rule::prior_day const& x) {
        rule.bookingType_ = api::BookingTypeEnum::UP_TO_PRIOR_DAYS;
        if (auto const last = days_before_service_day(x.prior_notice_last_day_,
                                                      x.prior_notice_bitfield_);
            last.has_value()) {
          rule.latestBookingTime_ =
              local_time_to_unixtime(sd->tz_, *last, x.prior_notice_last_time_);
        }
        if (x.prior_notice_start_day_ !=
            n::booking_rule::prior_day{}.prior_notice_start_day_) {
          if (auto const start = days_before_service_day(
                  x.prior_notice_start_day_, x.prior_notice_bitfield_);
              start.has_value()) {
            rule.earliestBookingTime_ = local_time_to_unixtime(
                sd->tz_, *start, x.prior_notice_start_time_);
          }
        }
      }});
  rule.message_ = translate(r.message_);
  rule.pickupMessage_ = translate(r.pickup_message_);
  rule.dropOffMessage_ = translate(r.drop_off_message_);
  rule.phoneNumber_ = translate(r.phone_number_);
  rule.infoUrl_ = translate(r.info_url_);
  rule.bookingUrl_ = translate(r.booking_url_);
  return rule;
}

}  // namespace

std::string_view get_flex_stop_name(n::timetable const& tt,
                                    n::lang_t const& lang,
                                    n::flex_stop_t const& s) {
  return s.apply(
      utl::overloaded{[&](n::flex_area_idx_t const a) {
                        return tt.translate(lang, tt.flex_area_name_[a]);
                      },
                      [&](n::location_group_idx_t const lg) {
                        return tt.translate(lang, tt.location_group_name_[lg]);
                      }});
}

std::string_view get_flex_id(n::timetable const& tt, n::flex_stop_t const& s) {
  return s.apply(utl::overloaded{[&](n::flex_area_idx_t const a) {
                                   return tt.strings_.get(tt.flex_area_id_[a]);
                                 },
                                 [&](n::location_group_idx_t const lg) {
                                   return tt.strings_.get(
                                       tt.location_group_id_[lg]);
                                 }});
}

flex_output::flex_output(osr::ways const& w,
                         osr::platforms const* pl,
                         platform_matches_t const* matches,
                         adr_ext const* ae,
                         tz_map_t const* tz,
                         tag_lookup const& tags,
                         n::timetable const& tt,
                         n::rt_timetable const* rtt,
                         flex_areas const& fa,
                         mode_payload const id,
                         flex_additional_nodes const& additional_nodes,
                         osr::sharing_data sharing_data)
    : w_{w},
      pl_{pl},
      matches_{matches},
      ae_{ae},
      tz_{tz},
      tt_{tt},
      rtt_{rtt},
      tags_{tags},
      fa_{fa},
      additional_nodes_{additional_nodes},
      sharing_data_{std::move(sharing_data)},
      mode_payload_{id} {}

flex_output::~flex_output() = default;

bool flex_output::is_time_dependent() const { return false; }

api::ModeEnum flex_output::get_mode() const { return api::ModeEnum::FLEX; }

transport_mode_t flex_output::get_cache_key() const {
  return transport_mode(api::ModeEnum::FLEX, mode_payload_.to_payload());
}

osr::search_profile flex_output::get_profile() const {
  return osr::search_profile::kCarSharing;
}

osr::sharing_data const* flex_output::get_sharing_data() const {
  return &sharing_data_;
}

void flex_output::annotate_leg(n::lang_t const& lang,
                               osr::node_idx_t const from,
                               osr::node_idx_t const to,
                               api::Leg& leg) const {
  if (from == osr::node_idx_t::invalid() || to == osr::node_idx_t::invalid()) {
    return;
  }

  auto const t = mode_payload_.get_flex_transport();
  auto const stop_seq = tt_.flex_stop_seq_[tt_.flex_transport_stop_seq_[t]];
  auto from_stop = std::optional<n::stop_idx_t>{};
  auto to_stop = std::optional<n::stop_idx_t>{};
  for (auto i = 0U; i != stop_seq.size(); ++i) {
    auto const stop_idx = static_cast<n::stop_idx_t>(
        mode_payload_.get_dir() == osr::direction::kForward
            ? i
            : stop_seq.size() - i - 1U);
    auto const stop = stop_seq[stop_idx];
    if (!from_stop.has_value() &&
        is_in_flex_stop(tt_, w_, fa_, additional_nodes_, stop, from)) {
      from_stop = stop_idx;
    } else if (!to_stop.has_value() &&
               is_in_flex_stop(tt_, w_, fa_, additional_nodes_, stop, to)) {
      to_stop = stop_idx;
      break;
    }
  }

  if (!from_stop.has_value()) {
    n::log(n::log_lvl::error, "flex", "flex: from  [node={}] not found", from);
    return;
  }

  if (!to_stop.has_value()) {
    n::log(n::log_lvl::error, "flex", "flex: to [node={}] not found", to);
    return;
  }

  auto const write_node_info = [&](api::Place& p, osr::node_idx_t const n) {
    if (w_.is_additional_node(n)) {
      auto const l = additional_nodes_.get(n);
      p = to_place(&tt_, &tags_, &w_, pl_, matches_, ae_, tz_, lang,
                   tt_location{l});
    }
  };
  write_node_info(leg.from_, from);
  write_node_info(leg.to_, to);

  leg.mode_ = api::ModeEnum::FLEX;
  leg.from_.flex_ = get_flex_stop_name(tt_, lang, stop_seq[*from_stop]);
  leg.from_.flexId_ = get_flex_id(tt_, stop_seq[*from_stop]);
  leg.to_.flex_ = get_flex_stop_name(tt_, lang, stop_seq[*to_stop]);
  leg.to_.flexId_ = get_flex_id(tt_, stop_seq[*to_stop]);

  auto const time_windows = tt_.flex_transport_stop_time_windows_[t];

  leg.from_.flexStartPickupDropOffWindow_ =
      std::chrono::time_point_cast<std::chrono::days>(leg.startTime_.time_) +
      time_windows[*from_stop].from_;
  leg.from_.flexEndPickupDropOffWindow_ =
      std::chrono::time_point_cast<std::chrono::days>(leg.startTime_.time_) +
      time_windows[*from_stop].to_;

  leg.to_.flexStartPickupDropOffWindow_ =
      std::chrono::time_point_cast<std::chrono::days>(leg.endTime_.time_) +
      time_windows[*to_stop].from_;
  leg.to_.flexEndPickupDropOffWindow_ =
      std::chrono::time_point_cast<std::chrono::days>(leg.endTime_.time_) +
      time_windows[*to_stop].to_;

  auto const sd =
      get_service_day(tt_, t, *from_stop,
                      std::chrono::time_point_cast<n::unixtime_t::duration>(
                          leg.startTime_.time_));
  leg.from_.pickupBookingRule_ = get_booking_rule(
      tt_, lang, tt_.flex_transport_pickup_booking_rule_[t][*from_stop], leg,
      sd);
  leg.to_.dropOffBookingRule_ = get_booking_rule(
      tt_, lang, tt_.flex_transport_drop_off_booking_rule_[t][*to_stop], leg,
      sd);

  if (rtt_ != nullptr) {
    auto const trip = tt_.flex_transport_trip_[t];
    auto alerts = std::vector<api::Alert>{};
    for (auto const& id : tt_.trip_ids_[trip]) {
      for (auto const& a :
           rtt_->alerts_.get_alerts(tt_, tt_.trip_id_src_[id], trip,
                                    n::rt_transport_idx_t::invalid(),
                                    n::location_idx_t::invalid(), false)) {
        alerts.emplace_back(to_alert(rtt_->alerts_, a, lang));
      }
    }
    if (!alerts.empty()) {
      leg.alerts_ = std::move(alerts);
    }
  }
}

api::Place flex_output::get_place(n::lang_t const& lang,
                                  osr::node_idx_t const n,
                                  std::optional<std::string> const& tz) const {
  if (w_.is_additional_node(n)) {
    auto const l = additional_nodes_.get(n);
    auto const c = tt_.locations_.coordinates_.at(l);
    return api::Place{
        .name_ = std::string{tt_.translate(lang, tt_.locations_.names_.at(l))},
        .lat_ = c.lat_,
        .lon_ = c.lng_,
        .tz_ = tz,
        .vertexType_ = api::VertexTypeEnum::TRANSIT};
  } else {
    auto const pos = w_.get_node_pos(n).as_latlng();
    return api::Place{.lat_ = pos.lat_,
                      .lon_ = pos.lng_,
                      .tz_ = tz,
                      .vertexType_ = api::VertexTypeEnum::NORMAL};
  }
}

}  // namespace motis::flex