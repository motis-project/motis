#pragma once

#include <cstdint>
#include <limits>
#include <optional>

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

// Walk offsets through a station entrance carry the entrance path in the
// payload: osr profile | (entrance path + 1) << kEntrancePathShift. The
// street search to the entrances is kept under entrance_search_mode().
constexpr auto const kEntrancePathShift = 8U;
constexpr auto const kProfileMask =
    transport_mode_t::payload_t{(1U << kEntrancePathShift) - 1U};
constexpr auto const kEntranceSearch =
    std::numeric_limits<transport_mode_t::payload_t>::max() >>
    kEntrancePathShift;

constexpr osr::search_profile profile_of(transport_mode_t const m) {
  return static_cast<osr::search_profile>(m.payload_ & kProfileMask);
}

constexpr auto const kClientOffsetPayload =
    std::numeric_limits<transport_mode_t::payload_t>::max();

constexpr transport_mode_t client_transport_mode(api::ModeEnum const m) {
  return transport_mode(m, kClientOffsetPayload);
}

constexpr bool is_client_offset(transport_mode_t const m) {
  return m.payload_ == kClientOffsetPayload;
}

constexpr transport_mode_t entrance_transport_mode(
    osr::search_profile const p, std::uint32_t const entrance_path) {
  return transport_mode(api::ModeEnum::WALK,
                        static_cast<transport_mode_t::payload_t>(p) |
                            ((entrance_path + 1U) << kEntrancePathShift));
}

constexpr transport_mode_t entrance_search_mode(osr::search_profile const p) {
  return transport_mode(api::ModeEnum::WALK,
                        static_cast<transport_mode_t::payload_t>(p) |
                            (kEntranceSearch << kEntrancePathShift));
}

constexpr std::optional<std::uint32_t> get_entrance_path(
    transport_mode_t const m) {
  if (to_mode(m) != api::ModeEnum::WALK || is_client_offset(m)) {
    return std::nullopt;
  }
  auto const x = m.payload_ >> kEntrancePathShift;
  return x == 0U || x == kEntranceSearch ? std::nullopt : std::optional{x - 1U};
}

// Routed as car, but rendered as themselves.
constexpr auto const kOdmTransportMode =
    transport_mode(api::ModeEnum::ODM, osr::search_profile::kCar);
constexpr auto const kRideSharingTransportMode =
    transport_mode(api::ModeEnum::RIDE_SHARING, osr::search_profile::kCar);

}  // namespace motis
