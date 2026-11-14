#pragma once

#include "utl/verify.h"

#include "net/bad_request_exception.h"

#include "motis-api/motis-api.h"

#include "nigiri/routing/transfer_time_settings.h"
#include "nigiri/types.h"

namespace motis {

template <typename Query>
bool use_routed_transfers(Query const& query) {
  return query.useRoutedTransfers_ || query.transferTimeFactor_ != 1.0 ||
         query.additionalTransferTime_ != 0;
}

inline nigiri::profile_idx_t get_profile_idx(
    bool const use_routed_transfers,
    bool const require_car_transport,
    api::PedestrianProfileEnum const pedestrian_profile) {
  if (!use_routed_transfers) {
    return nigiri::kDefaultProfile;
  }
  if (require_car_transport) {
    return nigiri::kCarProfile;
  }
  return pedestrian_profile == api::PedestrianProfileEnum::WHEELCHAIR
             ? nigiri::kWheelchairProfile
             : nigiri::kFootProfile;
}

template <typename Query>
nigiri::profile_idx_t get_profile_idx(Query const& query) {
  return get_profile_idx(use_routed_transfers(query),
                         query.requireCarTransport_, query.pedestrianProfile_);
}

template <typename Query>
nigiri::routing::transfer_time_settings get_transfer_time_settings(
    Query const& query) {
  utl::verify<net::bad_request_exception>(
      query.minTransferTime_ >= 0 && query.additionalTransferTime_ >= 0,
      "minTransferTime={} and additionalTransferTime={} must not be negative",
      query.minTransferTime_, query.additionalTransferTime_);
  return {.default_ = (query.minTransferTime_ == 0 &&
                       query.additionalTransferTime_ == 0 &&
                       query.transferTimeFactor_ == 1.0),
          .min_transfer_time_ = nigiri::duration_t{query.minTransferTime_},
          .additional_time_ = nigiri::duration_t{query.additionalTransferTime_},
          .factor_ = static_cast<float>(query.transferTimeFactor_)};
}

}  // namespace motis
