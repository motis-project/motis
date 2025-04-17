#pragma once

#include <cstdint>
#include <memory>

#include "motis/fwd.h"
#include "motis/gbfs/data.h"
#include "motis/types.h"

namespace motis::gbfs {

struct gbfs_routing_data {
  gbfs_routing_data() = default;
  gbfs_routing_data(osr::ways const* w,
                    osr::lookup const* l,
                    std::shared_ptr<gbfs_data> data)
      : w_{w}, l_{l}, data_{std::move(data)} {}

  bool has_data() const { return data_ != nullptr; }

  std::shared_ptr<provider_routing_data> get_provider_routing_data(
      gbfs_provider const&);

  products_routing_data* get_products_routing_data(
      gbfs_provider const& provider, gbfs_products_idx_t prod_idx);
  products_routing_data* get_products_routing_data(gbfs_products_ref);

  provider_products const& get_products(gbfs_products_ref);

  nigiri::transport_mode_id_t get_transport_mode(gbfs_products_ref);
  gbfs_products_ref get_products_ref(nigiri::transport_mode_id_t) const;

  osr::ways const* w_{};
  osr::lookup const* l_{};
  std::shared_ptr<gbfs_data> data_{};

  hash_map<gbfs_products_ref, std::shared_ptr<products_routing_data>> products_;
  std::vector<gbfs_products_ref> products_refs_;
  hash_map<gbfs_products_ref, nigiri::transport_mode_id_t>
      products_ref_to_transport_mode_;
};

std::shared_ptr<provider_routing_data> compute_provider_routing_data(
    osr::ways const&, osr::lookup const&, gbfs_provider const&);

std::shared_ptr<provider_routing_data> get_provider_routing_data(
    osr::ways const&, osr::lookup const&, gbfs_data&, gbfs_provider const&);

}  // namespace motis::gbfs
