#pragma once

#include <limits>

#include "osr/routing/profile.h"

#include "nigiri/routing/query.h"
#include "nigiri/types.h"

#include "motis-api/motis-api.h"
#include "motis/osr/mode_to_profile.h"

namespace motis {

using transport_mode_t = nigiri::routing::transport_mode_t;

constexpr transport_mode_t transport_mode(
    api::ModeEnum const m, transport_mode_t::payload_t const payload = 0U) {
  return {static_cast<transport_mode_t::mode_t>(m), payload};
}

constexpr transport_mode_t transport_mode(api::ModeEnum const m,
                                          osr::search_profile const p) {
  return transport_mode(m, static_cast<transport_mode_t::payload_t>(p));
}

inline transport_mode_t transport_mode(osr::search_profile const p) {
  return transport_mode(to_mode(p), p);
}

constexpr api::ModeEnum to_mode(transport_mode_t const m) {
  return static_cast<api::ModeEnum>(m.mode_);
}

constexpr osr::search_profile profile_of(transport_mode_t const m) {
  return static_cast<osr::search_profile>(m.payload_);
}

constexpr auto const kClientOffsetPayload =
    std::numeric_limits<transport_mode_t::payload_t>::max();

constexpr transport_mode_t client_transport_mode(api::ModeEnum const m) {
  return transport_mode(m, kClientOffsetPayload);
}

constexpr bool is_client_offset(transport_mode_t const m) {
  return m.payload_ == kClientOffsetPayload;
}

// Routed as car, but rendered as themselves.
constexpr auto const kOdmTransportMode =
    transport_mode(api::ModeEnum::ODM, osr::search_profile::kCar);
constexpr auto const kRideSharingTransportMode =
    transport_mode(api::ModeEnum::RIDE_SHARING, osr::search_profile::kCar);

}  // namespace motis
