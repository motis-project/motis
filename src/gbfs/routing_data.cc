#include "motis/gbfs/routing_data.h"

#include "osr/lookup.h"
#include "osr/types.h"
#include "osr/ways.h"

#include "fmt/format.h"

#include "utl/get_or_create.h"
#include "utl/timer.h"

#include "motis/constants.h"
#include "motis/gbfs/data.h"
#include "motis/gbfs/osr_mapping.h"
#include "motis/transport_mode.h"

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
  auto const ref = gbfs::gbfs_products_ref{provider.idx_, prod_idx};
  return utl::get_or_create(
             products_, ref,
             [&] { return data_->get_products_routing_data(*w_, *l_, ref); })
      .get();
}

products_routing_data* gbfs_routing_data::get_products_routing_data(
    gbfs_products_ref const prod_ref) {
  return get_products_routing_data(*data_->providers_.at(prod_ref.provider_),
                                   prod_ref.products_);
}

provider_products const& gbfs_routing_data::get_products(
    gbfs_products_ref const prod_ref) {
  return data_->providers_.at(prod_ref.provider_)
      ->products_.at(prod_ref.products_);
}

nigiri::routing::transport_mode_t gbfs_routing_data::get_transport_mode(
    gbfs_products_ref const prod_ref) {
  return transport_mode(
      api::ModeEnum::RENTAL,
      utl::get_or_create(products_ref_to_payload_, prod_ref, [&]() {
        auto const payload =
            static_cast<transport_mode_t::payload_t>(products_refs_.size());
        products_refs_.emplace_back(prod_ref);
        return payload;
      }));
}

gbfs_products_ref gbfs_routing_data::get_products_ref(
    nigiri::routing::transport_mode_t::payload_t const payload) const {
  return products_refs_.at(static_cast<std::size_t>(payload));
}

}  // namespace motis::gbfs
