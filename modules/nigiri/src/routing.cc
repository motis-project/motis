#include "motis/nigiri/routing.h"

#include "boost/thread/tss.hpp"

#include "opentelemetry/trace/scope.h"
#include "opentelemetry/trace/span.h"

#include "utl/erase_if.h"
#include "utl/helpers/algorithm.h"
#include "utl/pipes.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "nigiri/routing/limits.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/raptor/raptor_state.h"
#include "nigiri/routing/raptor_search.h"
#include "nigiri/routing/search.h"
#include "nigiri/special_stations.h"

#include "motis/core/common/timing.h"
#include "motis/core/access/error.h"
#include "motis/core/journey/journeys_to_message.h"
#include "motis/core/otel/tracer.h"
#include "motis/nigiri/location.h"
#include "motis/nigiri/metrics.h"
#include "motis/nigiri/nigiri_to_motis_journey.h"
#include "motis/nigiri/unixtime_conv.h"

namespace n = ::nigiri;
namespace mm = motis::module;
namespace fbs = flatbuffers;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
boost::thread_specific_ptr<n::routing::search_state> search_state;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
boost::thread_specific_ptr<n::routing::raptor_state> raptor_state;

namespace motis::nigiri {

mm::msg_ptr to_routing_response(
    n::timetable const& tt, n::rt_timetable const* rtt, tag_lookup const& tags,
    n::pareto_set<n::routing::journey> const* journeys,
    n::interval<n::unixtime_t> search_interval,
    n::routing::search_stats const& search_stats,
    n::routing::raptor_stats const& raptor_stats,
    std::uint64_t const routing_time) {
  mm::message_creator fbb;
  MOTIS_START_TIMING(conversion);
  auto const connections =
      utl::all(*journeys)  //
      | utl::remove_if([&](n::routing::journey const& j) {  //
          return j.error_;
        })  //
      | utl::transform([&](n::routing::journey const& j) {
          return to_connection(fbb, nigiri_to_motis_journey(tt, rtt, tags, j));
        })  //
      | utl::vec();
  MOTIS_STOP_TIMING(conversion);

  auto entries = std::vector<fbs::Offset<StatisticsEntry>>{
      CreateStatisticsEntry(fbb, fbb.CreateString("routing_time_ms"),
                            routing_time),
      CreateStatisticsEntry(fbb, fbb.CreateString("lower_bounds_time_ms"),
                            search_stats.lb_time_),
      CreateStatisticsEntry(fbb, fbb.CreateString("fastest_direct"),
                            search_stats.fastest_direct_),
      CreateStatisticsEntry(fbb, fbb.CreateString("footpaths_visited"),
                            raptor_stats.n_footpaths_visited_),
      CreateStatisticsEntry(fbb, fbb.CreateString("routes_visited"),
                            raptor_stats.n_routes_visited_),
      CreateStatisticsEntry(fbb, fbb.CreateString("earliest_trip_calls"),
                            raptor_stats.n_earliest_trip_calls_),
      CreateStatisticsEntry(
          fbb, fbb.CreateString("earliest_arrival_updated_by_route"),
          raptor_stats.n_earliest_arrival_updated_by_route_),
      CreateStatisticsEntry(
          fbb, fbb.CreateString("earliest_arrival_updated_by_footpath"),
          raptor_stats.n_earliest_arrival_updated_by_footpath_),
      CreateStatisticsEntry(
          fbb, fbb.CreateString("fp_update_prevented_by_lower_bound"),
          raptor_stats.fp_update_prevented_by_lower_bound_),
      CreateStatisticsEntry(
          fbb, fbb.CreateString("route_update_prevented_by_lower_bound"),
          raptor_stats.route_update_prevented_by_lower_bound_),
      CreateStatisticsEntry(fbb, fbb.CreateString("conversion"),
                            MOTIS_TIMING_MS(conversion))};
  auto statistics = std::vector<fbs::Offset<Statistics>>{
      CreateStatistics(fbb, fbb.CreateString("nigiri.raptor"),
                       fbb.CreateVectorOfSortedTables(&entries))};
  fbb.create_and_finish(
      MsgContent_RoutingResponse,
      routing::CreateRoutingResponse(
          fbb, fbb.CreateVectorOfSortedTables(&statistics),
          fbb.CreateVector(connections),
          to_motis_unixtime(search_interval.from_),
          to_motis_unixtime(search_interval.to_),
          fbb.CreateVector(std::vector<fbs::Offset<DirectConnection>>{}))
          .Union());
  return make_msg(fbb);
}

std::vector<n::routing::offset> get_offsets(
    tag_lookup const& tags, n::timetable const& tt,
    fbs::Vector<fbs::Offset<motis::routing::AdditionalEdgeWrapper>> const*
        edges,
    SearchDir const dir, bool const is_start) {
  auto const ref_station = n::get_special_station_name(
      is_start ? n::special_station::kStart : n::special_station::kEnd);
  return utl::all(*edges)  //
         | utl::transform([](routing::AdditionalEdgeWrapper const* e) {
             return reinterpret_cast<routing::MumoEdge const*>(
                 e->additional_edge());
           })  //
         | utl::remove_if([&](routing::MumoEdge const* e) {
             // XOR Table:
             // is_fwd     | is_start      | use_from
             // FWD  true  | START  true   | to_station    false
             // FWD  true  | END    false  | from_station  true
             // BWD  false | START  true   | from_station  true
             // BWD  false | END    false  | to_station    false
             auto const x = ((dir == SearchDir_Forward) ^ is_start) == 0U
                                ? e->from_station_id()->view()
                                : e->to_station_id()->view();
             return x != ref_station;
           })  //
         | utl::transform([&](routing::MumoEdge const* e) {
             return n::routing::offset{
                 get_location_idx(tags, tt,
                                  ((dir == SearchDir_Forward) ^ is_start) == 0U
                                      ? e->to_station_id()->str()
                                      : e->from_station_id()->str()),
                 n::duration_t{static_cast<std::int16_t>(e->duration())},
                 e->mumo_id()};
           })  //
         | utl::vec();
}

n::routing::clasz_mask_t to_clasz_mask(fbs::Vector<std::uint8_t> const* v) {
  if (v == nullptr) {
    return n::routing::all_clasz_allowed();
  } else {
    auto mask = n::routing::clasz_mask_t{0U};
    for (auto const c : *v) {
      utl::verify(c < static_cast<std::uint8_t>(n::clasz::kNumClasses),
                  "clasz {} does not exist", c);
      mask |= (1U << c);
    }
    return mask;
  }
}

motis::module::msg_ptr route(tag_lookup const& tags, n::timetable const& tt,
                             n::rt_timetable const* rtt,
                             motis::module::msg_ptr const& msg,
                             metrics& metrics, n::profile_idx_t const prf_idx) {
  using motis::routing::RoutingRequest;
  auto const req = motis_content(RoutingRequest, msg);

  auto span = motis_tracer->StartSpan("nigiri::route");
  auto scope = opentelemetry::trace::Scope{span};

  auto min_connection_count = static_cast<std::uint8_t>(0U);
  auto extend_interval_earlier = false;
  auto extend_interval_later = false;
  auto start_time = n::routing::start_time_t{};
  auto start_station = n::location_idx_t::invalid();
  auto timeout = [&]() -> std::optional<std::chrono::seconds> {
    if (req->timeout() == 0) {
      return std::nullopt;
    } else {
      return {std::chrono::seconds(req->timeout())};
    }
  }();

  if (req->start_type() == routing::Start_PretripStart) {
    metrics.pretrip_requests_.Increment();
    auto const start =
        reinterpret_cast<routing::PretripStart const*>(req->start());
    start_time = n::interval<n::unixtime_t>{
        to_nigiri_unixtime(start->interval()->begin()),
        to_nigiri_unixtime(start->interval()->end()) + std::chrono::minutes{1}};
    start_station = get_location_idx(tags, tt, start->station()->id()->view());
    min_connection_count = start->min_connection_count();
    extend_interval_earlier = start->extend_interval_earlier();
    extend_interval_later = start->extend_interval_later();
  } else if (req->start_type() == routing::Start_OntripStationStart) {
    metrics.ontrip_station_requests_.Increment();
    auto const start =
        reinterpret_cast<routing::OntripStationStart const*>(req->start());
    start_time = to_nigiri_unixtime(start->departure_time());
    start_station = get_location_idx(tags, tt, start->station()->id()->view());
    utl::verify(start_station != n::location_idx_t::invalid(),
                "unknown station {}", start->station()->id()->view());
  } else {
    throw utl::fail("OntripTrainStart not supported");
  }
  metrics.via_count_.Observe(req->via()->size());
  auto const destination_station =
      get_location_idx(tags, tt, req->destination()->id()->str());

  std::visit(
      utl::overloaded{
          [&](n::unixtime_t const t) {
            if (!tt.external_interval().contains(t)) {
              throw std::system_error(access::error::timestamp_not_in_schedule);
            }
          },
          [&](n::interval<n::unixtime_t>& interval) {
            auto const tt_interval = tt.external_interval();
            if (!interval.overlaps(tt_interval)) {
              throw std::system_error(access::error::timestamp_not_in_schedule);
            }
            interval.from_ = tt_interval.clamp(interval.from_);
            interval.to_ = tt_interval.clamp(interval.to_);
          }},
      start_time);

  auto const is_intermodal_start =
      start_station == n::get_special_station(n::special_station::kStart);
  auto const is_intermodal_dest =
      destination_station == n::get_special_station(n::special_station::kEnd);

  for (auto const& e : *req->additional_edges()) {
    utl::verify(e->additional_edge_type() == routing::AdditionalEdge_MumoEdge,
                "not a mumo edge: {}",
                routing::EnumNameAdditionalEdge(e->additional_edge_type()));
  }

  for (auto const& e : *req->additional_edges()) {
    auto const me =
        reinterpret_cast<routing::MumoEdge const*>(e->additional_edge());
    auto const a = req->search_dir() == SearchDir_Forward
                       ? me->from_station_id()->view()
                       : me->to_station_id()->view();
    auto const b = req->search_dir() == SearchDir_Backward
                       ? me->from_station_id()->view()
                       : me->to_station_id()->view();
    utl::verify(
        (is_intermodal_dest &&
         b == n::get_special_station_name(n::special_station::kEnd)) ||
            (is_intermodal_start &&
             a == n::get_special_station_name(n::special_station::kStart)) ||
            (is_intermodal_start && is_intermodal_dest &&
             a == n::get_special_station_name(n::special_station::kStart) &&
             b == n::get_special_station_name(n::special_station::kEnd)),
        "bad mumo edge: {} -> {}", me->from_station_id()->view(),
        me->to_station_id()->view());
  }

  utl::verify(!is_intermodal_start ||
                  utl::any_of(*req->additional_edges(),
                              [&](routing::AdditionalEdgeWrapper const* e) {
                                auto const me =
                                    reinterpret_cast<routing::MumoEdge const*>(
                                        e->additional_edge());
                                auto const x =
                                    req->search_dir() == SearchDir_Forward
                                        ? me->from_station_id()->view()
                                        : me->to_station_id()->view();
                                return x == n::get_special_station_name(
                                                n::special_station::kStart);
                              }),
              "intermodal start but no edge from START");

  utl::verify(!is_intermodal_dest ||
                  utl::any_of(*req->additional_edges(),
                              [&](routing::AdditionalEdgeWrapper const* e) {
                                auto const me =
                                    reinterpret_cast<routing::MumoEdge const*>(
                                        e->additional_edge());
                                auto const x =
                                    req->search_dir() == SearchDir_Forward
                                        ? me->to_station_id()->view()
                                        : me->from_station_id()->view();
                                return x == n::get_special_station_name(
                                                n::special_station::kEnd);
                              }),
              "intermodal destination but no edge to END");

  utl::verify(destination_station != n::location_idx_t::invalid(),
              "unknown station {}", req->destination()->id()->view());

  auto destination = is_intermodal_dest
                         ? get_offsets(tags, tt, req->additional_edges(),
                                       req->search_dir(), false)
                         : std::vector<n::routing::offset>{
                               {destination_station, n::duration_t{0U}, 0U}};

  auto max_transfers = n::routing::kMaxTransfers;
  if (req->max_transfers() >= 0) {
    utl::verify(req->max_transfers() < ::nigiri::routing::kMaxTransfers,
                "unsupported max_transfers value (max supported: {})",
                ::nigiri::routing::kMaxTransfers - 1);
    max_transfers = static_cast<std::uint8_t>(req->max_transfers() + 1);
  }

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
                                  req->search_dir(), true)
                    : std::vector<n::routing::offset>{{start_station,
                                                       n::duration_t{0U}, 0U}},
      .destination_ = std::move(destination),
      .max_transfers_ = max_transfers,
      .min_connection_count_ = min_connection_count,
      .extend_interval_earlier_ = extend_interval_earlier,
      .extend_interval_later_ = extend_interval_later,
      .prf_idx_ = prf_idx,
      .allowed_claszes_ = to_clasz_mask(req->allowed_claszes()),
      .require_bike_transport_ = req->bike_transport(),
      .transfer_time_settings_ = n::routing::transfer_time_settings{
          .default_ = req->min_transfer_time() <= 0 &&
                      req->transfer_time_factor() == 1.0F,
          .min_transfer_time_ = n::duration_t{req->min_transfer_time()},
          .factor_ = req->transfer_time_factor()}};

  for (auto const& via : *req->via()) {
    auto const station =
        get_location_idx(tags, tt, via->station()->id()->view());
    utl::verify(station != n::location_idx_t::invalid(),
                "unknown via station {}", via->station()->id()->view());
    q.via_stops_.emplace_back(n::routing::via_stop{
        .location_ = station, .stay_ = n::duration_t{via->stay_duration()}});
  }

  utl::verify(!q.start_.empty(), "no start edges");
  utl::verify(!q.destination_.empty(), "no destination edges");

  if (search_state.get() == nullptr) {
    search_state.reset(new n::routing::search_state{});
  }
  if (raptor_state.get() == nullptr) {
    raptor_state.reset(new n::routing::raptor_state{});
  }

  MOTIS_START_TIMING(routing);
  auto search_interval = n::interval<n::unixtime_t>{};
  n::pareto_set<n::routing::journey> const* journeys{nullptr};
  n::routing::search_stats search_stats;
  n::routing::raptor_stats raptor_stats;
  auto const dir = req->search_dir() == SearchDir_Forward
                       ? n::direction::kForward
                       : n::direction::kBackward;
  auto const r = n::routing::raptor_search(
      tt, rtt, *search_state, *raptor_state, std::move(q), dir, timeout);
  journeys = r.journeys_;
  search_stats = r.search_stats_;
  raptor_stats = r.algo_stats_;
  search_interval = r.interval_;
  MOTIS_STOP_TIMING(routing);

  auto const reconstruction_errors = static_cast<std::int64_t>(utl::count_if(
      *r.journeys_, [](n::routing::journey const& j) { return j.error_; }));
  metrics.reconstruction_errors_.Observe(reconstruction_errors);

  if (req->start_type() == routing::Start_PretripStart) {
    metrics.pretrip_routing_time_.Observe(MOTIS_TIMING_S(routing));
    metrics.pretrip_interval_extensions_.Observe(
        static_cast<double>(search_stats.interval_extensions_));
  } else if (req->start_type() == routing::Start_OntripStationStart) {
    metrics.ontrip_station_routing_time_.Observe(MOTIS_TIMING_S(routing));
  }

  span->AddEvent("routing done",
                 {{"journeys", r.journeys_->size()},
                  {"reconstruction_errors", reconstruction_errors}});

  span->SetAttribute("motis.nigiri.result.journeys",
                     static_cast<std::int64_t>(r.journeys_->size()));

  span->SetAttribute("motis.nigiri.result.reconstruction_errors",
                     reconstruction_errors);

  return to_routing_response(tt, rtt, tags, journeys, search_interval,
                             search_stats, raptor_stats,
                             MOTIS_TIMING_MS(routing));
}

}  // namespace motis::nigiri
