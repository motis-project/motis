#pragma once

#include "motis/street_routing.h"

namespace motis::gbfs {

struct gbfs_output final : public output {
  gbfs_output(osr::ways const&,
              gbfs::gbfs_routing_data&,
              gbfs::gbfs_products_ref);
  ~gbfs_output() override;

  transport_mode_t get_cache_key(osr::search_profile) const override;

  osr::sharing_data const* get_sharing_data() const override;

  api::VertexTypeEnum get_vertex_type() const override;

  void annotate(osr::node_idx_t const from_node,
                osr::node_idx_t const to_node,
                api::Leg& leg) const override;

  geo::latlng get_node_pos(osr::node_idx_t const n) const override;

  std::string get_node_name(osr::node_idx_t const n) const override;

  std::size_t get_additional_node_idx(osr::node_idx_t const n) const;

  osr::ways const& w_;
  gbfs::gbfs_routing_data& gbfs_rd_;
  gbfs::gbfs_provider const& provider_;
  gbfs::provider_products const& products_;
  gbfs::products_routing_data const* prod_rd_;
  osr::sharing_data sharing_data_;
  api::Rental rental_;
};

}  // namespace motis::gbfs