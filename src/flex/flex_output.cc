#include "motis/flex/flex_output.h"

#include "nigiri/flex.h"
#include "nigiri/timetable.h"

#include "motis/flex/flex.h"
#include "motis/flex/flex_areas.h"
#include "motis/flex/flex_routing_data.h"
#include "motis/osr/street_routing.h"
#include "motis/place.h"

namespace n = nigiri;

namespace motis::flex {

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
  auto const from_stop = mode_payload_.get_from_stop();
  auto const to_stop = mode_payload_.get_to_stop();

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
  leg.from_.flex_ = get_flex_stop_name(tt_, lang, stop_seq[from_stop]);
  leg.from_.flexId_ = get_flex_id(tt_, stop_seq[from_stop]);
  leg.to_.flex_ = get_flex_stop_name(tt_, lang, stop_seq[to_stop]);
  leg.to_.flexId_ = get_flex_id(tt_, stop_seq[to_stop]);

  // Windows of the service day the ride belongs to, not of the calendar day
  // of the leg (differs for windows past midnight).
  set_flex_windows(
      tt_, mode_payload_,
      get_service_day(tt_, mode_payload_, leg.startTime_.time_), leg);
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