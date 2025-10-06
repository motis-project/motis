#include "motis/odm/odm.h"

#include <ranges>

#include "nigiri/routing/journey.h"

#include "motis/transport_mode_ids.h"

namespace motis::odm {

namespace n = nigiri;
namespace nr = nigiri::routing;

bool by_stop(nr::start const& a, nr::start const& b) {
  return std::tie(a.stop_, a.time_at_start_, a.time_at_stop_) <
         std::tie(b.stop_, b.time_at_start_, b.time_at_stop_);
}

bool is_odm_leg(nr::journey::leg const& l,
                nigiri::transport_mode_id_t const mode) {
  return std::holds_alternative<nr::offset>(l.uses_) &&
         std::get<nr::offset>(l.uses_).transport_mode_id_ == mode;
}

bool uses_odm(nr::journey const& j, nigiri::transport_mode_id_t const mode) {
  return utl::any_of(j.legs_,
                     [&](auto const& l) { return is_odm_leg(l, mode); });
}

bool is_pure_pt(nr::journey const& j) {
  return !uses_odm(j, kOdmTransportModeId) &&
         !uses_odm(j, kRideSharingTransportModeId);
};

n::duration_t odm_time(nr::journey::leg const& l) {
  return is_odm_leg(l, kOdmTransportModeId) ||
                 is_odm_leg(l, kRideSharingTransportModeId)
             ? std::get<nr::offset>(l.uses_).duration()
             : n::duration_t{0};
}

n::duration_t odm_time(nr::journey const& j) {
  return std::transform_reduce(begin(j.legs_), end(j.legs_), n::duration_t{0},
                               std::plus{},
                               [](auto const& l) { return odm_time(l); });
}

n::duration_t pt_time(nr::journey const& j) {
  return j.travel_time() - odm_time(j);
}

bool is_direct_odm(nr::journey const& j) {
  return j.travel_time() == odm_time(j);
}

n::duration_t duration(nr::start const& ride) {
  return std::chrono::abs(ride.time_at_stop_ - ride.time_at_start_);
}

std::string odm_label(nr::journey const& j) {
  return fmt::format(
      "[dep: {}, arr: {}, transfers: {}, start_odm: {}, dest_odm: {}]",
      j.start_time_, j.dest_time_, j.transfers_, odm_time(j.legs_.front()),
      odm_time(j.legs_.back()));
}

}  // namespace motis::odm