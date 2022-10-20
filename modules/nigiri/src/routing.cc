#include "motis/nigiri/routing.h"

#include "boost/thread/tss.hpp"

#include "utl/to_vec.h"
#include "utl/verify.h"

#include "nigiri/routing/limits.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/raptor.h"
#include "nigiri/routing/search_state.h"

#include "motis/core/common/timing.h"
#include "motis/core/journey/journeys_to_message.h"
#include "motis/nigiri/location.h"
#include "motis/nigiri/nigiri_to_motis_journey.h"
#include "motis/nigiri/unixtime_conv.h"

namespace n = ::nigiri;
namespace mm = motis::module;
namespace fbs = flatbuffers;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
boost::thread_specific_ptr<n::routing::search_state> search_state;

namespace motis::nigiri {

mm::msg_ptr to_routing_response(
    n::timetable const& tt, std::vector<std::string> const& tags,
    n::pareto_set<n::routing::journey> const& journeys,
    n::interval<n::unixtime_t> search_interval,
    n::routing::stats const& stats) {
  mm::message_creator fbb;
  MOTIS_START_TIMING(conversion);
  auto const connections =
      utl::to_vec(journeys, [&](n::routing::journey const& j) {
        return to_connection(fbb, nigiri_to_motis_journey(tt, tags, j));
      });
  MOTIS_STOP_TIMING(conversion);
  std::vector<fbs::Offset<Statistics>> statistics{CreateStatistics(
      fbb, fbb.CreateString("nigiri.raptor"),
      fbb.CreateVector(std::vector<fbs::Offset<StatisticsEntry>>{
          CreateStatisticsEntry(fbb, fbb.CreateString("routing_time_ms"),
                                stats.n_routing_time_),
          CreateStatisticsEntry(fbb, fbb.CreateString("lower_bounds_time_ms"),
                                stats.lb_time_),
          CreateStatisticsEntry(fbb, fbb.CreateString("footpaths_visited"),
                                stats.n_footpaths_visited_),
          CreateStatisticsEntry(fbb, fbb.CreateString("routes_visited"),
                                stats.n_routes_visited_),
          CreateStatisticsEntry(fbb, fbb.CreateString("earliest_trip_calls"),
                                stats.n_earliest_trip_calls_),
          CreateStatisticsEntry(
              fbb, fbb.CreateString("earliest_arrival_updated_by_route"),
              stats.n_earliest_arrival_updated_by_route_),
          CreateStatisticsEntry(
              fbb, fbb.CreateString("earliest_arrival_updated_by_footpath"),
              stats.n_earliest_arrival_updated_by_footpath_),
          CreateStatisticsEntry(
              fbb, fbb.CreateString("fp_update_prevented_by_lower_bound"),
              stats.fp_update_prevented_by_lower_bound_),
          CreateStatisticsEntry(
              fbb, fbb.CreateString("route_update_prevented_by_lower_bound"),
              stats.route_update_prevented_by_lower_bound_),
          CreateStatisticsEntry(fbb, fbb.CreateString("conversion"),
                                MOTIS_TIMING_MS(conversion))}))};
  fbb.create_and_finish(
      MsgContent_RoutingResponse,
      routing::CreateRoutingResponse(
          fbb, fbb.CreateVectorOfSortedTables(&statistics),
          fbb.CreateVector(connections),
          to_motis_unixtime(search_interval.from_),
          to_motis_unixtime(search_interval.to_ - std::chrono::minutes{1}),
          fbb.CreateVector(std::vector<fbs::Offset<DirectConnection>>{}))
          .Union());
  return make_msg(fbb);
}

motis::module::msg_ptr route(std::vector<std::string> const& tags,
                             n::timetable& tt,
                             motis::module::msg_ptr const& msg) {
  using motis::routing::RoutingRequest;
  auto const req = motis_content(RoutingRequest, msg);

  auto start_time = n::routing::start_time_t{};
  auto start_station = n::location_idx_t::invalid();
  if (req->start_type() == routing::Start_PretripStart) {
    auto const start =
        reinterpret_cast<routing::PretripStart const*>(req->start());
    utl::verify(start->min_connection_count() == 0U &&
                    !start->extend_interval_earlier() &&
                    !start->extend_interval_later(),
                "nigiri currently does not support interval extension");
    start_time = n::interval<n::unixtime_t>{
        to_nigiri_unixtime(start->interval()->begin()),
        to_nigiri_unixtime(start->interval()->end()) + std::chrono::minutes{1}};
    start_station = get_location_idx(tags, tt, start->station()->id()->str());
  } else if (req->start_type() == routing::Start_OntripStationStart) {
    auto const start =
        reinterpret_cast<routing::OntripStationStart const*>(req->start());
    start_time = to_nigiri_unixtime(start->departure_time());
    start_station = get_location_idx(tags, tt, start->station()->id()->str());
    utl::verify(start_station != n::location_idx_t::invalid(),
                "unknown station {}", start->station()->id()->c_str());
  } else {
    throw utl::fail("OntripTrainStart not supported");
  }

  auto const destination_station =
      get_location_idx(tags, tt, req->destination()->id()->str());
  utl::verify(destination_station != n::location_idx_t::invalid(),
              "unknown station {}", req->destination()->id()->c_str());

  auto q = n::routing::query{
      .start_time_ = start_time,
      .start_match_mode_ = req->use_start_metas()
                               ? n::routing::location_match_mode::kEquivalent
                               : n::routing::location_match_mode::kOnlyChildren,
      .dest_match_mode_ = req->use_dest_metas()
                              ? n::routing::location_match_mode::kEquivalent
                              : n::routing::location_match_mode::kOnlyChildren,
      .use_start_footpaths_ = req->use_start_footpaths(),
      .start_ = {n::routing::offset{.location_ = start_station,
                                    .offset_ = n::duration_t{0U},
                                    .type_ = 0U}},
      .destinations_ = {std::vector<n::routing::offset>{
          n::routing::offset{.location_ = destination_station,
                             .offset_ = n::duration_t{0U},
                             .type_ = 0U}}},
      .via_destinations_ = {},
      .allowed_classes_ = cista::bitset<n::kNumClasses>::max(),
      .max_transfers_ = n::routing::kMaxTransfers,
      .min_connection_count_ = 0U,
      .extend_interval_earlier_ = false,
      .extend_interval_later_ = false};

  if (search_state.get() == nullptr) {
    search_state.reset(new n::routing::search_state{});
  }

  n::routing::stats stats;
  MOTIS_START_TIMING(routing);
  if (req->search_dir() == SearchDir_Forward) {
    auto r = n::routing::raptor<n::direction::kForward>{tt, *search_state,
                                                        std::move(q)};
    r.route();
    stats = r.get_stats();
  } else {
    auto r = n::routing::raptor<n::direction::kBackward>{tt, *search_state,
                                                         std::move(q)};
    r.route();
    stats = r.get_stats();
  }
  MOTIS_STOP_TIMING(routing);
  stats.n_routing_time_ = MOTIS_TIMING_MS(routing);

  return to_routing_response(tt, tags, search_state->results_.at(0),
                             search_state->search_interval_, stats);
}

}  // namespace motis::nigiri
