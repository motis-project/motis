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
    auto timer = utl::scoped_timer{
        fmt::format("compute routing data for gbfs provider {}", provider.id_)};
    return compute_provider_routing_data(w, l, provider);
  });
}

std::shared_ptr<provider_routing_data>
gbfs_routing_data::get_provider_routing_data(gbfs_provider const& provider) {
  return gbfs::get_provider_routing_data(*w_, *l_, *data_, provider);
}

segment_routing_data* gbfs_routing_data::get_segment_routing_data(
    gbfs_provider const& provider, gbfs_segment_idx_t const seg_idx) {
  auto const seg_ref = gbfs::gbfs_segment_ref{provider.idx_, seg_idx};
  if (auto const it = segments_.find(seg_ref); it != end(segments_)) {
    return it->second.get();
  } else {
    return segments_
        .emplace(seg_ref,
                 get_provider_routing_data(provider)->get_segment_routing_data(
                     seg_idx))
        .first->second.get();
  }
}

segment_routing_data* gbfs_routing_data::get_segment_routing_data(
    gbfs_segment_ref const seg_ref) {
  return get_segment_routing_data(*data_->providers_.at(seg_ref.provider_),
                                  seg_ref.segment_);
}

nigiri::transport_mode_id_t gbfs_routing_data::get_transport_mode(
    gbfs_segment_ref const seg_ref) {
  if (auto const it = segment_ref_to_transport_mode_.find(seg_ref);
      it != end(segment_ref_to_transport_mode_)) {
    return it->second;
  } else {
    auto const id = static_cast<nigiri::transport_mode_id_t>(
        kGbfsTransportModeIdOffset + segment_refs_.size());
    segment_refs_.emplace_back(seg_ref);
    segment_ref_to_transport_mode_[seg_ref] = id;
    return id;
  }
}

gbfs_segment_ref gbfs_routing_data::get_segment_ref(
    nigiri::transport_mode_id_t const id) const {
  return segment_refs_.at(id - kGbfsTransportModeIdOffset);
}

}  // namespace motis::gbfs
