#pragma once

#include <optional>

#include "osr/routing/profile.h"

#include "motis-api/motis-api.h"

#include "motis/fwd.h"
#include "motis/osr/street_routing.h"
#include "motis/transport_mode.h"

namespace motis {

struct default_output final : public output {
  default_output(osr::ways const&, osr::search_profile);
  default_output(osr::ways const&, transport_mode_t);
  ~default_output() override;

  bool is_time_dependent() const override;
  api::ModeEnum get_mode() const override;
  osr::search_profile get_profile() const override;
  transport_mode_t get_cache_key() const override;
  osr::sharing_data const* get_sharing_data() const override;
  void annotate_leg(nigiri::lang_t const&,
                    osr::node_idx_t,
                    osr::node_idx_t,
                    api::Leg&) const override;
  api::Place get_place(nigiri::lang_t const&,
                       osr::node_idx_t,
                       std::optional<std::string> const& tz) const override;

  osr::ways const& w_;
  osr::search_profile profile_;
  transport_mode_t id_;
};

}  // namespace motis
