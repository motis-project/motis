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

  segment_routing_data* get_segment_routing_data(gbfs_provider const& provider,
                                                 gbfs_segment_idx_t seg_idx);
  segment_routing_data* get_segment_routing_data(gbfs_segment_ref);

  nigiri::transport_mode_id_t get_transport_mode(gbfs_segment_ref);
  gbfs_segment_ref get_segment_ref(nigiri::transport_mode_id_t) const;

  osr::ways const* w_{};
  osr::lookup const* l_{};
  std::shared_ptr<gbfs_data> data_{};

  hash_map<gbfs_segment_ref, std::shared_ptr<segment_routing_data>> segments_;
  std::vector<gbfs_segment_ref> segment_refs_;
  hash_map<gbfs_segment_ref, nigiri::transport_mode_id_t>
      segment_ref_to_transport_mode_;
};

std::shared_ptr<provider_routing_data> compute_provider_routing_data(
    osr::ways const&, osr::lookup const&, gbfs_provider const&);

std::shared_ptr<provider_routing_data> get_provider_routing_data(
    osr::ways const&, osr::lookup const&, gbfs_data&, gbfs_provider const&);

}  // namespace motis::gbfs
