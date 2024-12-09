#include "motis/gbfs/data.h"

#include "osr/lookup.h"
#include "osr/ways.h"

#include "motis/gbfs/compression.h"
#include "motis/gbfs/routing_data.h"

namespace motis::gbfs {

products_routing_data::products_routing_data(
    std::shared_ptr<provider_routing_data const>&& prd,
    compressed_routing_data const& compressed)
    : provider_routing_data_{std::move(prd)}, compressed_{compressed} {
  decompress_bitvec(compressed_.start_allowed_, start_allowed_);
  decompress_bitvec(compressed_.end_allowed_, end_allowed_);
  decompress_bitvec(compressed_.through_allowed_, through_allowed_);
}

std::shared_ptr<products_routing_data> gbfs_data::get_products_routing_data(
    osr::ways const& w,
    osr::lookup const& l,
    gbfs_products_ref const prod_ref) {
  auto lock = std::unique_lock{products_routing_data_mutex_};

  if (auto it = products_routing_data_.find(prod_ref);
      it != end(products_routing_data_)) {
    if (auto prod_rd = it->second.lock(); prod_rd) {
      return prod_rd;
    }
  }

  auto provider_rd = get_provider_routing_data(
      w, l, *this, *providers_.at(prod_ref.provider_));
  auto prod_rd = provider_rd->get_products_routing_data(prod_ref.products_);
  products_routing_data_[prod_ref] = prod_rd;
  return prod_rd;
}

}  // namespace motis::gbfs
