#include "motis/flex/flex_output.h"

#include "nigiri/timetable.h"

#include "motis/flex/flex.h"
#include "motis/flex/flex_routing_data.h"
#include "motis/street_routing.h"

namespace n = nigiri;

namespace motis::flex {

flex_output::flex_output(osr::ways const& w,
                         osr::lookup const& l,
                         osr::platforms const* pl,
                         platform_matches_t const* matches,
                         n::timetable const& tt,
                         mode_id const id)
    : w_{w},
      tt_{tt},
      sharing_data_{flex::prepare_sharing_data(
          tt, w, l, pl, matches, id, id.get_dir(), flex_routing_data_)},
      mode_id_(id) {}

flex_output::~flex_output() = default;

api::ModeEnum flex_output::get_mode() const { return api::ModeEnum::FLEX; }

transport_mode_t flex_output::get_cache_key() const { return mode_id_.to_id(); }

osr::search_profile flex_output::get_profile() const {
  return osr::search_profile::kCarSharing;
}

osr::sharing_data const* flex_output::get_sharing_data() const {
  return &sharing_data_;
}

void flex_output::annotate_leg(osr::node_idx_t,
                               osr::node_idx_t,
                               api::Leg&) const {}

api::Place flex_output::get_place(osr::node_idx_t const n) const {
  if (w_.is_additional_node(n)) {
    auto const l =
        flex_routing_data_.additional_nodes_.at(get_additional_node_idx(n));
    auto const c = tt_.locations_.coordinates_.at(l);
    return api::Place{.name_ = std::string{tt_.locations_.names_.at(l).view()},
                      .lat_ = c.lat_,
                      .lon_ = c.lng_,
                      .vertexType_ = api::VertexTypeEnum::TRANSIT};
  } else {
    auto const pos = w_.get_node_pos(n).as_latlng();
    return api::Place{.lat_ = pos.lat_,
                      .lon_ = pos.lng_,
                      .vertexType_ = api::VertexTypeEnum::NORMAL};
  }
}

std::size_t flex_output::get_additional_node_idx(
    osr::node_idx_t const n) const {
  return to_idx(n) - sharing_data_.additional_node_offset_;
}

}  // namespace motis::flex