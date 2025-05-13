#include "motis/flex/flex_output.h"

#include "nigiri/timetable.h"

#include "motis/flex/flex.h"
#include "motis/flex/flex_routing_data.h"
#include "motis/street_routing.h"

namespace n = nigiri;

namespace motis::flex {

flex_output::flex_output(
    osr::ways const& w,
    osr::lookup const& l,
    osr::platforms const* pl,
    platform_matches_t const* matches,
    n::timetable const& tt,
    n::flex_stop_t from_stop,
    std::vector<n::flex_stop_t> const& to_stops,
    std::vector<std::pair<n::flex_transport_idx_t, n::stop_idx_t>> const&)
    : w_{w},
      tt_{tt},
      sharing_data_{flex::prepare_sharing_data(
          tt, w, l, pl, matches, from_stop, to_stops, flex_routing_data_)} {}

api::VertexTypeEnum flex_output::get_vertex_type() const {
  return api::VertexTypeEnum::TRANSIT;
}

osr::sharing_data const* flex_output::get_sharing_data() const {
  return &sharing_data_;
}

std::string flex_output::get_node_name(osr::node_idx_t const n) const {
  return std::string{
      tt_.locations_
          .names_[flex_routing_data_
                      .additional_nodes_[get_additional_node_idx(n)]]
          .view()};
}

geo::latlng flex_output::get_node_pos(osr::node_idx_t const n) const {
  return tt_.locations_.coordinates_
      [flex_routing_data_.additional_nodes_[get_additional_node_idx(n)]];
}

void flex_output::annotate(osr::node_idx_t, osr::node_idx_t, api::Leg&) const {}

std::size_t flex_output::get_additional_node_idx(
    osr::node_idx_t const n) const {
  return to_idx(n) - sharing_data_.additional_node_offset_;
}

}  // namespace motis::flex