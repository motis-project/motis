#include "motis/osr/default_output.h"

#include "osr/routing/sharing_data.h"

#include "motis/osr/mode_to_profile.h"

namespace n = nigiri;

namespace motis {

default_output::default_output(osr::ways const& w,
                               osr::search_profile const profile)
    : default_output{w, transport_mode(profile)} {}

default_output::default_output(osr::ways const& w, transport_mode_t const mode)
    : w_{w}, profile_{profile_of(mode)}, mode_{mode} {}

default_output::~default_output() = default;

api::ModeEnum default_output::get_mode() const { return to_mode(mode_); }

osr::search_profile default_output::get_profile() const { return profile_; }

api::Place default_output::get_place(
    nigiri::lang_t const&,
    osr::node_idx_t const n,
    std::optional<std::string> const& tz) const {
  auto const pos = w_.get_node_pos(n).as_latlng();
  return api::Place{.lat_ = pos.lat_,
                    .lon_ = pos.lng_,
                    .tz_ = tz,
                    .vertexType_ = api::VertexTypeEnum::NORMAL};
}

bool default_output::is_time_dependent() const {
  return profile_ == osr::search_profile::kWheelchair ||
         profile_ == osr::search_profile::kHgv ||
         profile_ == osr::search_profile::kCarParkingWheelchair ||
         profile_ == osr::search_profile::kCarDropOffWheelchair;
}

transport_mode_t default_output::get_cache_key() const { return mode_; }

osr::sharing_data const* default_output::get_sharing_data() const {
  return nullptr;
}

void default_output::annotate_leg(n::lang_t const&,
                                  osr::node_idx_t,
                                  osr::node_idx_t,
                                  api::Leg&) const {}

}  // namespace motis
