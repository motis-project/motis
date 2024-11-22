#include "motis/gbfs/data.h"

#include "motis/gbfs/compression.h"

namespace motis::gbfs {

segment_routing_data::segment_routing_data(
    std::shared_ptr<provider_routing_data const>&& prd,
    compressed_routing_data const& compressed)
    : provider_routing_data_{std::move(prd)}, compressed_{compressed} {
  decompress_bitvec(compressed_.start_allowed_, start_allowed_);
  decompress_bitvec(compressed_.end_allowed_, end_allowed_);
  decompress_bitvec(compressed_.through_allowed_, through_allowed_);
}

}  // namespace motis::gbfs
