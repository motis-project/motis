#pragma once

#include "motis/flex/flex_routing_data.h"
#include "motis/flex/mode_id.h"
#include "motis/street_routing.h"

namespace motis::flex {

std::string_view get_flex_stop_name(nigiri::timetable const&,
                                    nigiri::flex_stop_t const&);

std::string_view get_flex_id(nigiri::timetable const&,
                             nigiri::flex_stop_t const&);

struct flex_output : public output {
  flex_output(osr::ways const&,
              osr::lookup const&,
              osr::platforms const*,
              platform_matches_t const*,
              tag_lookup const&,
              nigiri::timetable const&,
              flex_areas const&,
              mode_id);
  ~flex_output() override;

  api::ModeEnum get_mode() const override;
  osr::search_profile get_profile() const override;
  bool is_time_dependent() const override;
  transport_mode_t get_cache_key() const override;
  osr::sharing_data const* get_sharing_data() const override;
  void annotate_leg(osr::node_idx_t, osr::node_idx_t, api::Leg&) const override;
  api::Place get_place(osr::node_idx_t) const override;

  std::size_t get_additional_node_idx(osr::node_idx_t) const;

private:
  osr::ways const& w_;
  osr::platforms const* pl_;
  platform_matches_t const* matches_;
  nigiri::timetable const& tt_;
  tag_lookup const& tags_;
  flex_areas const& fa_;
  flex::flex_routing_data flex_routing_data_;
  osr::sharing_data sharing_data_;
  mode_id mode_id_;
};

}  // namespace motis::flex