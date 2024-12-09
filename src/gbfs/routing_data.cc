#include "motis/gbfs/routing_data.h"

#include "osr/lookup.h"
#include "osr/types.h"
#include "osr/ways.h"

#include "fmt/format.h"

#include "utl/timer.h"

#include "motis/constants.h"
#include "motis/gbfs/data.h"
#include "motis/gbfs/osr_mapping.h"

namespace motis::gbfs {

std::shared_ptr<provider_routing_data> compute_provider_routing_data(
    osr::ways const& w, osr::lookup const& l, gbfs_provider const& provider) {
  auto timer = utl::scoped_timer{
      fmt::format("compute routing data for gbfs provider {}", provider.id_)};
  auto prd = std::make_shared<provider_routing_data>();

  map_data(w, l, provider, *prd);

  return prd;
}

std::shared_ptr<provider_routing_data> get_provider_routing_data(
    osr::ways const& w,
    osr::lookup const& l,
    gbfs_data& data,
    gbfs_provider const& provider) {
  return data.cache_.get_or_compute(provider.idx_, [&]() {
    return compute_provider_routing_data(w, l, provider);
  });
}

std::shared_ptr<provider_routing_data>
gbfs_routing_data::get_provider_routing_data(gbfs_provider const& provider) {
  return gbfs::get_provider_routing_data(*w_, *l_, *data_, provider);
}

products_routing_data* gbfs_routing_data::get_products_routing_data(
    gbfs_provider const& provider, gbfs_products_idx_t const prod_idx) {
  auto const prod_ref = gbfs::gbfs_products_ref{provider.idx_, prod_idx};
  if (auto const it = products_.find(prod_ref); it != end(products_)) {
    return it->second.get();
  } else {
    return products_
        .emplace(prod_ref, data_->get_products_routing_data(*w_, *l_, prod_ref))
        .first->second.get();
  }
}

products_routing_data* gbfs_routing_data::get_products_routing_data(
    gbfs_products_ref const prod_ref) {
  return get_products_routing_data(*data_->providers_.at(prod_ref.provider_),
                                   prod_ref.products_);
}

nigiri::transport_mode_id_t gbfs_routing_data::get_transport_mode(
    gbfs_products_ref const prod_ref) {
  if (auto const it = products_ref_to_transport_mode_.find(prod_ref);
      it != end(products_ref_to_transport_mode_)) {
    return it->second;
  } else {
    auto const id = static_cast<nigiri::transport_mode_id_t>(
        kGbfsTransportModeIdOffset + products_refs_.size());
    products_refs_.emplace_back(prod_ref);
    products_ref_to_transport_mode_[prod_ref] = id;
    return id;
  }
}

gbfs_products_ref gbfs_routing_data::get_products_ref(
    nigiri::transport_mode_id_t const id) const {
  return products_refs_.at(
      static_cast<std::size_t>(id - kGbfsTransportModeIdOffset));
}

}  // namespace motis::gbfs
