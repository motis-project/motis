#pragma once

#include "motis/osr/street_routing.h"

namespace motis::gbfs {

struct gbfs_output final : public output {
  gbfs_output(osr::ways const&,
              gbfs::gbfs_routing_data&,
              gbfs::gbfs_products_ref,
              bool ignore_rental_return_constraints);
  ~gbfs_output() override;

  api::ModeEnum get_mode() const override;

  osr::search_profile get_profile() const override;

  bool is_time_dependent() const override;
  transport_mode_t get_cache_key() const override;

  osr::sharing_data const* get_sharing_data() const override;

  void annotate_leg(nigiri::lang_t const&,
                    osr::node_idx_t const from_node,
                    osr::node_idx_t const to_node,
                    api::Leg&) const override;

  api::Place get_place(nigiri::lang_t const&,
                       osr::node_idx_t,
                       std::optional<std::string> const& tz) const override;

  std::size_t get_additional_node_idx(osr::node_idx_t const n) const;

private:
  std::string get_node_name(osr::node_idx_t) const;

  osr::ways const& w_;
  gbfs::gbfs_routing_data& gbfs_rd_;
  gbfs::gbfs_provider const& provider_;
  gbfs::provider_products const& products_;
  gbfs::products_routing_data const* prod_rd_;
  osr::sharing_data sharing_data_;
  api::Rental rental_;
};

}  // namespace motis::gbfs
