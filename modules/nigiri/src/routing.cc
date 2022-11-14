#include "motis/nigiri/routing.h"

#include "boost/thread/tss.hpp"

#include "utl/helpers/algorithm.h"
#include "utl/pipes.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "nigiri/routing/limits.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/raptor.h"
#include "nigiri/routing/search_state.h"
#include "nigiri/special_stations.h"

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

std::vector<n::routing::offset> get_offsets(
    std::vector<std::string> const& tags, n::timetable const& tt,
    fbs::Vector<fbs::Offset<motis::routing::AdditionalEdgeWrapper>> const*
        edges,
    n::direction const search_dir) {
  return utl::all(*edges)  //
         | utl::transform([](routing::AdditionalEdgeWrapper const* e) {
             return reinterpret_cast<routing::MumoEdge const*>(
                 e->additional_edge());
           })  //
         | utl::remove_if([&](routing::MumoEdge const* e) {
             if (search_dir == n::direction::kForward) {
               return e->from_station_id()->view() !=
                      n::get_special_station_name(n::special_station::kStart);
             } else {
               return e->to_station_id()->view() !=
                      n::get_special_station_name(n::special_station::kEnd);
             }
           })  //
         | utl::transform([&](routing::MumoEdge const* e) {
             return n::routing::offset{
                 get_location_idx(tags, tt,
                                  search_dir == n::direction::kForward
                                      ? e->to_station_id()->str()
                                      : e->from_station_id()->str()),

                 n::duration_t{static_cast<std::int16_t>(e->duration())},
                 static_cast<std::uint8_t>(e->mumo_id())};
           })  //
         | utl::vec();
}

template <n::direction SearchDir, bool IntermodalDest>
n::routing::stats run_search(n::routing::search_state& state,
                             n::timetable const& tt, n::routing::query&& q) {
  n::routing::stats stats;
  auto r = n::routing::raptor<n::direction::kForward, IntermodalDest>{
      tt, state, std::move(q)};
  r.route();
  stats = r.get_stats();
  return stats;
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

  auto const is_intermodal_start =
      start_station == n::get_special_station(n::special_station::kStart);
  auto const is_intermodal_dest =
      destination_station == n::get_special_station(n::special_station::kEnd);

  utl::verify(
      utl::all_of(
          *req->additional_edges(),
          [&](routing::AdditionalEdgeWrapper const* e) {
            if (e->additional_edge_type() != routing::AdditionalEdge_MumoEdge) {
              return false;
            }
            auto const mumo_e =
                static_cast<routing::MumoEdge const*>(e->additional_edge());
            return (is_intermodal_start &&
                    mumo_e->from_station_id()->view() == "START") ||
                   (is_intermodal_dest &&
                    mumo_e->to_station_id()->view() == "END");
          }),
      "nigiri only supports mumo edges to end or from start");

  utl::verify(destination_station != n::location_idx_t::invalid(),
              "unknown station {}", req->destination()->id()->view());

  auto q = n::routing::query{
      .start_time_ = start_time,
      .start_match_mode_ = is_intermodal_start
                               ? n::routing::location_match_mode::kIntermodal
                           : req->use_start_metas()
                               ? n::routing::location_match_mode::kEquivalent
                               : n::routing::location_match_mode::kOnlyChildren,
      .dest_match_mode_ = is_intermodal_dest
                              ? n::routing::location_match_mode::kIntermodal
                          : req->use_dest_metas()
                              ? n::routing::location_match_mode::kEquivalent
                              : n::routing::location_match_mode::kOnlyChildren,
      .use_start_footpaths_ = req->use_start_footpaths(),
      .start_ = is_intermodal_start
                    ? get_offsets(tags, tt, req->additional_edges(),
                                  n::direction::kForward)
                    : std::vector<n::routing::offset>{{start_station,
                                                       n::duration_t{0U}, 0U}},
      .destinations_ =
          std::vector{
              {is_intermodal_start
                   ? get_offsets(tags, tt, req->additional_edges(),
                                 n::direction::kBackward)
                   : std::vector<n::routing::offset>{{destination_station,
                                                      n::duration_t{0U}, 0U}}}},
      .via_destinations_ = {},
      .allowed_classes_ = cista::bitset<n::kNumClasses>::max(),
      .max_transfers_ = n::routing::kMaxTransfers,
      .min_connection_count_ = 0U,
      .extend_interval_earlier_ = false,
      .extend_interval_later_ = false};

  utl::verify(!q.start_.empty(), "no start edges");
  utl::verify(!q.destinations_[0].empty(), "no destination edges");

  if (search_state.get() == nullptr) {
    search_state.reset(new n::routing::search_state{});
  }

  MOTIS_START_TIMING(routing);
  n::routing::stats stats;
  if (req->search_dir() == SearchDir_Forward) {
    if (is_intermodal_dest) {
      stats = run_search<n::direction::kForward, true>(*search_state, tt,
                                                       std::move(q));
    } else {
      stats = run_search<n::direction::kForward, false>(*search_state, tt,
                                                        std::move(q));
    }
  } else {
    if (is_intermodal_dest) {
      stats = run_search<n::direction::kBackward, true>(*search_state, tt,
                                                        std::move(q));
    } else {
      stats = run_search<n::direction::kBackward, false>(*search_state, tt,
                                                         std::move(q));
    }
  }
  MOTIS_STOP_TIMING(routing);
  stats.n_routing_time_ = MOTIS_TIMING_MS(routing);

  return to_routing_response(tt, tags, search_state->results_.at(0),
                             search_state->search_interval_, stats);
}

}  // namespace motis::nigiri
