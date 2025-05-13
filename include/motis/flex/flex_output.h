#pragma once

#include "motis/flex/flex_routing_data.h"
#include "motis/street_routing.h"

namespace motis::flex {

struct flex_output : public output {
  flex_output(osr::ways const&,
              osr::lookup const&,
              osr::platforms const*,
              platform_matches_t const*,
              nigiri::timetable const&,
              nigiri::flex_stop_t from_stop,
              std::vector<nigiri::flex_stop_t> const& to_stops,
              std::vector<std::pair<nigiri::flex_transport_idx_t,
                                    nigiri::stop_idx_t>> const&);
  ~flex_output() override;

  transport_mode_t get_cache_key() const override;
  api::VertexTypeEnum get_vertex_type() const override;
  osr::sharing_data const* get_sharing_data() const override;
  std::string get_node_name(osr::node_idx_t const n) const override;
  geo::latlng get_node_pos(osr::node_idx_t const n) const override;
  void annotate_leg(osr::node_idx_t, osr::node_idx_t, api::Leg&) const override;
  void annotate_place(osr::);

private:
  std::size_t get_additional_node_idx(osr::node_idx_t const n) const;

  osr::ways const& w_;
  nigiri::timetable const& tt_;
  flex::flex_routing_data flex_routing_data_;
  osr::sharing_data sharing_data_;
};

}  // namespace motis::flex