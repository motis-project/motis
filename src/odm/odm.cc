#include "motis/odm/odm.h"

#include "boost/fiber/fss.hpp"

#include "nigiri/routing/journey.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/raptor/raptor_state.h"
#include "nigiri/routing/raptor_search.h"

#include "motis-api/motis-api.h"
#include "motis/endpoints/routing.h"
#include "motis/place.h"

namespace motis::ep {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static boost::fibers::fiber_specific_ptr<nigiri::routing::search_state>
    search_state;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static boost::fibers::fiber_specific_ptr<nigiri::routing::raptor_state>
    raptor_state;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static boost::fibers::fiber_specific_ptr<osr::bitvec<osr::node_idx_t>> blocked;

std::optional<std::vector<nigiri::routing::journey>> odm_routing(
    routing const& r,
    api::plan_params const& query,
    std::vector<api::ModeEnum> const& pre_transit_modes,
    std::vector<api::ModeEnum> const& post_transit_modes,
    std::vector<api::ModeEnum> const& direct_modes,
    std::variant<osr::location, tt_location> const& from,
    std::variant<osr::location, tt_location> const& to,
    api::Place const& from_p,
    api::Place const& to_p,
    std::variant<osr::location, tt_location> const& start,
    std::variant<osr::location, tt_location> const& dest,
    std::vector<api::ModeEnum> const& start_modes,
    std::vector<api::ModeEnum> const& dest_modes,
    nigiri::routing::query const& start_time,
    std::optional<nigiri::unixtime_t> const& t) {
  auto const rt = r.rt_;
  auto const rtt = rt->rtt_.get();
  auto const e = r.rt_->e_.get();
  auto const gbfs = r.gbfs_;

  auto const odm_pre_transit =
      std::find(begin(pre_transit_modes), end(pre_transit_modes),
                api::ModeEnum::ODM) != end(pre_transit_modes);
  auto const odm_post_transit =
      std::find(begin(post_transit_modes), end(post_transit_modes),
                api::ModeEnum::ODM) != end(post_transit_modes);
  auto const odm_start = query.arriveBy_ ? odm_post_transit : odm_pre_transit;
  auto const odm_dest = query.arriveBy_ ? odm_pre_transit : odm_post_transit;
  auto const odm_direct = std::find(begin(direct_modes), end(direct_modes),
                                    api::ModeEnum::ODM) != end(direct_modes);
  auto const odm_any = odm_pre_transit || odm_post_transit || odm_direct;
  auto odm_stats = motis::ep::stats_map_t{};

  if (!odm_any) {
    return std::nullopt;
  }

  auto const odm_start_offsets =
      odm_start && holds_alternative<osr::location>(start)
          ? r.get_offsets(std::get<osr::location>(start),
                          query.arriveBy_ ? osr::direction::kBackward
                                          : osr::direction::kForward,
                          {api::ModeEnum::CAR}, query.wheelchair_,
                          std::chrono::seconds{query.maxPreTransitTime_},
                          query.maxMatchingDistance_, gbfs.get(), odm_stats)
          : std::vector<nigiri::routing::offset>{};
  auto const odm_dest_offsets =
      odm_dest && holds_alternative<osr::location>(dest)
          ? r.get_offsets(std::get<osr::location>(dest),
                          query.arriveBy_ ? osr::direction::kForward
                                          : osr::direction::kBackward,
                          {api::ModeEnum::CAR}, query.wheelchair_,
                          std::chrono::seconds{query.maxPostTransitTime_},
                          query.maxMatchingDistance_, gbfs.get(), odm_stats)
          : std::vector<nigiri::routing::offset>{};

  // TODO collect departures/arrivals for each offset
  auto start_events = std::vector<std::vector<nigiri::unixtime_t>>{};
  start_events.resize(odm_start_offsets.size());

  // TODO ODM direct

  // TODO blacklist request

  // TODO remove blacklisted offsets

  // TODO start fibers to do the ODM routing

  // TODO whitelist request for ODM rides used in journeys

  // TODO remove journeys with non-whitelisted ODM rides

  return std::vector<nigiri::routing::journey>{};
}

}  // namespace motis::ep