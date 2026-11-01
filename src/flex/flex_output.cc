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

namespace {

// `n` days before `day`. If `service` is set, only its days are counted
// (GTFS `prior_notice_service_id`).
std::optional<date::local_days> days_before(n::timetable const& tt,
                                            date::local_days day,
                                            unsigned n,
                                            n::bitfield_idx_t const service) {
  if (service == n::bitfield_idx_t::invalid()) {
    return day - date::days{n};
  }
  auto const first =
      date::local_days{tt.internal_interval_days().from_.time_since_epoch()};
  while (n != 0U) {
    day -= date::days{1};
    if (day < first) {
      return std::nullopt;
    }
    if (tt.bitfields_[service].test(
            static_cast<std::size_t>((day - first).count()))) {
      --n;
    }
  }
  return day;
}

std::optional<api::BookingRule> get_booking_rule(
    n::timetable const& tt,
    n::lang_t const& lang,
    n::booking_rule_idx_t const idx,
    n::unixtime_t const start,
    date::time_zone const* tz) {
  if (idx == n::booking_rule_idx_t::invalid()) {
    return std::nullopt;
  }

  auto const translate =
      [&](n::translation_idx_t const t) -> std::optional<std::string> {
    auto const s = tt.translate(lang, t);
    return s.empty() ? std::nullopt : std::optional{std::string{s}};
  };

  // Local time on the day `n` days before the day of travel.
  auto const day_of_travel = date::floor<date::days>(tz->to_local(start));
  auto const local_time =
      [&](unsigned const n, n::duration_t const time,
          n::bitfield_idx_t const service) -> std::optional<n::unixtime_t> {
    auto const day = days_before(tt, day_of_travel, n, service);
    if (!day.has_value()) {
      return std::nullopt;
    }
    return std::chrono::time_point_cast<n::unixtime_t::duration>(
        tz->to_sys(*day + time, date::choose::earliest));
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
        rule.latestBookingTime_ = start - x.prior_notice_duration_min_;
        if (x.prior_notice_duration_max_ !=
            n::booking_rule::prior_notice{}.prior_notice_duration_max_) {
          rule.priorNoticeDurationMax_ = x.prior_notice_duration_max_.count();
          rule.earliestBookingTime_ = start - x.prior_notice_duration_max_;
        }
      },
      [&](n::booking_rule::prior_day const& x) {
        rule.bookingType_ = api::BookingTypeEnum::UP_TO_PRIOR_DAYS;
        rule.latestBookingTime_ =
            local_time(x.prior_notice_last_day_, x.prior_notice_last_time_,
                       x.prior_notice_bitfield_);
        if (x.prior_notice_start_day_ !=
            n::booking_rule::prior_day{}.prior_notice_start_day_) {
          rule.earliestBookingTime_ =
              local_time(x.prior_notice_start_day_, x.prior_notice_start_time_,
                         x.prior_notice_bitfield_);
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

  auto const trip = tt_.flex_transport_trip_[t];
  auto const get_alerts =
      [&](n::location_idx_t const l) -> std::optional<std::vector<api::Alert>> {
    if (rtt_ == nullptr) {
      return std::nullopt;
    }
    auto alerts = std::vector<api::Alert>{};
    for (auto const& a :
         rtt_->alerts_.get_alerts(tt_, tt_.trip_id_src_[trip], trip,
                                  n::rt_transport_idx_t::invalid(), l, false)) {
      alerts.emplace_back(to_alert(rtt_->alerts_, a, lang));
    }
    return alerts.empty() ? std::nullopt : std::optional{std::move(alerts)};
  };

  auto const write_node_info = [&](api::Place& p, osr::node_idx_t const n) {
    if (w_.is_additional_node(n)) {
      auto const l = additional_nodes_.get(n);
      p = to_place(&tt_, &tags_, &w_, pl_, matches_, ae_, tz_, lang,
                   tt_location{l});
      p.alerts_ = get_alerts(l);
    }
  };
  write_node_info(leg.from_, from);
  write_node_info(leg.to_, to);

  leg.mode_ = api::ModeEnum::FLEX;
  leg.from_.flex_ = get_flex_stop_name(tt_, lang, stop_seq[*from_stop]);
  leg.from_.flexId_ = get_flex_id(tt_, stop_seq[*from_stop]);
  leg.to_.flex_ = get_flex_stop_name(tt_, lang, stop_seq[*to_stop]);
  leg.to_.flexId_ = get_flex_id(tt_, stop_seq[*to_stop]);

  // Time windows are relative to the (UTC) traffic day of the transport.
  auto const time_windows = tt_.flex_transport_stop_time_windows_[t];
  auto const day = std::chrono::floor<std::chrono::days>(
      leg.startTime_.time_ - time_windows[*from_stop].from_);

  leg.from_.flexStartPickupDropOffWindow_ =
      day + time_windows[*from_stop].from_;
  leg.from_.flexEndPickupDropOffWindow_ = day + time_windows[*from_stop].to_;
  leg.to_.flexStartPickupDropOffWindow_ = day + time_windows[*to_stop].from_;
  leg.to_.flexEndPickupDropOffWindow_ = day + time_windows[*to_stop].to_;

  auto const src = tt_.trip_id_src_[trip];
  auto const agency =
      tt_.route_ids_[src].route_id_provider_[tt_.trip_route_id_[trip]];
  auto const tz = n::to_time_zone(tt_.timezones_[tt_.providers_[agency].tz_]);
  auto const start =
      std::chrono::time_point_cast<n::unixtime_t::duration>(*leg.startTime_);
  leg.from_.pickupBookingRule_ = get_booking_rule(
      tt_, lang, tt_.flex_transport_pickup_booking_rule_[t][*from_stop], start,
      tz);
  leg.to_.dropOffBookingRule_ = get_booking_rule(
      tt_, lang, tt_.flex_transport_drop_off_booking_rule_[t][*to_stop], start,
      tz);

  leg.alerts_ = get_alerts(n::location_idx_t::invalid());
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