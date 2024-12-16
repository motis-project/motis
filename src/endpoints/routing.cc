#include "motis/endpoints/routing.h"

#include "boost/fiber/future.hpp"
#include "boost/thread/tss.hpp"

#include "utl/erase_duplicates.h"
#include "utl/helpers/algorithm.h"

#include "osr/platforms.h"
#include "osr/routing/profiles/foot.h"
#include "osr/routing/route.h"
#include "osr/routing/sharing_data.h"

#include "nigiri/routing/limits.h"
#include "nigiri/routing/pareto_set.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/raptor/raptor_state.h"
#include "nigiri/routing/raptor_search.h"
#include "nigiri/special_stations.h"

#include "motis/constants.h"
#include "motis/endpoints/routing.h"
#include "motis/gbfs/data.h"
#include "motis/journey_to_response.h"
#include "motis/max_distance.h"
#include "motis/mode_to_profile.h"
#include "motis/odm/odm.h"
#include "motis/parse_location.h"
#include "motis/street_routing.h"
#include "motis/tag_lookup.h"
#include "motis/timetable/modes_to_clasz_mask.h"
#include "motis/timetable/time_conv.h"
#include "motis/update_rtt_td_footpaths.h"

namespace n = nigiri;
using namespace std::chrono_literals;

namespace motis::ep {

constexpr auto const kInfinityDuration =
    n::duration_t{std::numeric_limits<n::duration_t::rep>::max()};

using td_offsets_t =
    n::hash_map<n::location_idx_t, std::vector<n::routing::td_offset>>;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
boost::thread_specific_ptr<n::routing::search_state> search_state;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
boost::thread_specific_ptr<n::routing::raptor_state> raptor_state;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
boost::thread_specific_ptr<osr::bitvec<osr::node_idx_t>> blocked;

stats_map_t join(auto&&... maps) {
  auto ret = std::map<std::string, std::uint64_t>{};
  auto const add = [&](std::map<std::string, std::uint64_t> const& x) {
    ret.insert(begin(x), end(x));
  };
  (add(maps), ...);
  return ret;
}

place_t get_place(n::timetable const* tt,
                  tag_lookup const* tags,
                  std::string_view s) {
  if (auto const location = parse_location(s); location.has_value()) {
    return *location;
  }
  utl::verify(tt != nullptr && tags != nullptr,
              R"(could not parse location (no timetable loaded): "{}")", s);
  return tt_location{tags->get_location(*tt, s)};
}

bool is_intermodal(place_t const& p) {
  return std::holds_alternative<osr::location>(p);
}

n::routing::location_match_mode get_match_mode(place_t const& p) {
  return is_intermodal(p) ? n::routing::location_match_mode::kIntermodal
                          : n::routing::location_match_mode::kEquivalent;
}

std::vector<n::routing::offset> station_start(n::location_idx_t const l) {
  return {{l, n::duration_t{0U}, 0U}};
}

td_offsets_t routing::get_td_offsets(elevators const& e,
                                     osr::location const& pos,
                                     osr::direction const dir,
                                     std::vector<api::ModeEnum> const& modes,
                                     bool const wheelchair,
                                     std::chrono::seconds const max) const {
  if (!w_ || !l_ || !pl_ || !tt_ || !loc_tree_ || !matches_) {
    return {};
  }

  auto ret = hash_map<n::location_idx_t, std::vector<n::routing::td_offset>>{};
  for (auto const m : modes) {
    auto const profile = to_profile(m, wheelchair);

    if (profile != osr::search_profile::kWheelchair) {
      continue;  // handled by get_offsets
    }

    utl::equal_ranges_linear(
        get_td_footpaths(*w_, *l_, *pl_, *tt_, *loc_tree_, e, *matches_,
                         n::location_idx_t::invalid(), pos, dir, profile, max,
                         *blocked),
        [](n::td_footpath const& a, n::td_footpath const& b) {
          return a.target_ == b.target_;
        },
        [&](auto&& from, auto&& to) {
          ret.emplace(from->target_,
                      utl::to_vec(from, to, [&](n::td_footpath const fp) {
                        return n::routing::td_offset{
                            .valid_from_ = fp.valid_from_,
                            .duration_ = fp.duration_,
                            .transport_mode_id_ =
                                static_cast<n::transport_mode_id_t>(profile)};
                      }));
        });
  }

  return ret;
}

std::vector<n::routing::offset> routing::get_offsets(
    osr::location const& pos,
    osr::direction const dir,
    std::vector<api::ModeEnum> const& modes,
    bool const wheelchair,
    std::chrono::seconds const max,
    unsigned const max_matching_distance,
    gbfs::gbfs_data const* gbfs,
    stats_map_t& stats) const {
  if (!loc_tree_ || !pl_ || !tt_ || !loc_tree_ || !matches_) {
    return {};
  }

  auto offsets = std::vector<n::routing::offset>{};
  auto ignore_walk = false;

  auto const handle_mode = [&](api::ModeEnum const m) {
    auto const profile = to_profile(m, wheelchair);

    if (rt_->e_ && profile == osr::search_profile::kWheelchair) {
      return;  // handled by get_td_offsets
    }

    auto const max_dist = get_max_distance(profile, max);

    auto const near_stops = loc_tree_->in_radius(pos.pos_, max_dist);
    auto const near_stop_locations =
        utl::to_vec(near_stops, [&](n::location_idx_t const l) {
          return osr::location{tt_->locations_.coordinates_[l],
                               pl_->get_level(*w_, (*matches_)[l])};
        });

    if (profile == osr::search_profile::kBikeSharing) {
      if (gbfs == nullptr) {
        return;
      }

      auto providers = hash_set<gbfs_provider_idx_t>{};
      gbfs->provider_rtree_.in_radius(
          pos.pos_, max_dist, [&](auto const pi) { providers.insert(pi); });

      for (auto const& pi : providers) {
        UTL_START_TIMING(timer);
        auto const& provider = gbfs->providers_.at(pi);
        auto const sharing = provider->get_sharing_data(w_->n_nodes());
        auto const paths =
            osr::route(*w_, *l_, profile, pos, near_stop_locations,
                       static_cast<osr::cost_t>(max.count()), dir,
                       kMaxMatchingDistance, nullptr, &sharing);
        ignore_walk = true;
        for (auto const [p, l] : utl::zip(paths.paths_, near_stops)) {
          if (p.has_value()) {
            offsets.emplace_back(l, n::duration_t{p->cost_ / 60},
                                 static_cast<n::transport_mode_id_t>(
                                     kGbfsTransportModeIdOffset + to_idx(pi)));
          }
        }
        UTL_STOP_TIMING(timer);
        stats.emplace(fmt::format("GBFS_{}_{}_{}", provider->id_,
                                  fmt::streamed(dir), fmt::streamed(m)),
                      UTL_TIMING_MS(timer));
        stats.emplace(fmt::format("GBFS_{}_{}_{}_lookup", provider->id_,
                                  fmt::streamed(dir), fmt::streamed(m)),
                      paths.lookup_time_.count());
      }

    } else {
      UTL_START_TIMING(timer);
      auto const paths = osr::route(*w_, *l_, profile, pos, near_stop_locations,
                                    static_cast<osr::cost_t>(max.count()), dir,
                                    max_matching_distance, nullptr, nullptr);
      for (auto const [p, l] : utl::zip(paths.paths_, near_stops)) {
        if (p.has_value()) {
          offsets.emplace_back(l, n::duration_t{p->cost_ / 60},
                               static_cast<n::transport_mode_id_t>(profile));
        }
      }
      UTL_STOP_TIMING(timer);
      stats.emplace(fmt::format("{}_{}", fmt::streamed(dir), fmt::streamed(m)),
                    UTL_TIMING_MS(timer));
      stats.emplace(
          fmt::format("{}_{}_lookup", fmt::streamed(dir), fmt::streamed(m)),
          paths.lookup_time_.count());
    }
  };

  if (utl::find(modes, api::ModeEnum::BIKE_RENTAL) != end(modes)) {
    handle_mode(api::ModeEnum::BIKE_RENTAL);
  }

  for (auto const m : modes) {
    if (m == api::ModeEnum::BIKE_RENTAL) {
      continue;  // handled above
    }
    if (m == api::ModeEnum::WALK && ignore_walk) {
      continue;
    }
    handle_mode(m);
  }

  return offsets;
}

std::pair<n::routing::query, std::optional<n::unixtime_t>> get_start_time(
    api::plan_params const& query) {
  if (query.pageCursor_.has_value()) {
    return {cursor_to_query(*query.pageCursor_), std::nullopt};
  } else {
    auto const t = std::chrono::time_point_cast<n::i32_minutes>(
        *query.time_.value_or(openapi::now()));
    auto const window = std::chrono::duration_cast<n::duration_t>(
        std::chrono::seconds{query.searchWindow_ * (query.arriveBy_ ? -1 : 1)});
    return {{.start_time_ = query.timetableView_
                                ? n::routing::start_time_t{n::interval{
                                      query.arriveBy_ ? t - window : t,
                                      query.arriveBy_ ? t : t + window}}
                                : n::routing::start_time_t{t},
             .extend_interval_earlier_ = query.arriveBy_,
             .extend_interval_later_ = !query.arriveBy_},
            t};
  }
}

std::pair<std::vector<api::Itinerary>, n::duration_t> routing::route_direct(
    elevators const* e,
    gbfs::gbfs_data const* gbfs,
    api::Place const& from,
    api::Place const& to,
    std::vector<api::ModeEnum> const& modes,
    n::unixtime_t const start_time,
    bool wheelchair,
    std::chrono::seconds max) const {
  if (!w_ || !l_) {
    return {};
  }
  auto const omit_walk =
      gbfs != nullptr &&
      utl::find(modes, api::ModeEnum::BIKE_RENTAL) != end(modes);
  auto fastest_direct = kInfinityDuration;
  auto cache = street_routing_cache_t{};
  auto itineraries = std::vector<api::Itinerary>{};
  for (auto const& m : modes) {
    if (m == api::ModeEnum::CAR || m == api::ModeEnum::BIKE ||
        m == api::ModeEnum::CAR_PARKING ||
        (!omit_walk && m == api::ModeEnum::WALK)) {
      auto itinerary = route(
          *w_, *l_, gbfs, e, from, to, m, wheelchair, start_time, std::nullopt,
          gbfs_provider_idx_t::invalid(), cache, blocked.get(), max);
      if (itinerary.legs_.empty()) {
        continue;
      }
      auto const duration = std::chrono::duration_cast<n::duration_t>(
          std::chrono::seconds{itinerary.duration_});
      if (duration < fastest_direct) {
        fastest_direct = duration;
      }
      itineraries.emplace_back(std::move(itinerary));
    } else if (m == api::ModeEnum::BIKE_RENTAL && gbfs != nullptr) {
      auto const max_dist =
          get_max_distance(osr::search_profile::kBikeSharing, max);
      auto providers = hash_set<gbfs_provider_idx_t>{};
      gbfs->provider_rtree_.in_radius(
          {from.lat_, from.lon_}, max_dist,
          [&](auto const pi) { providers.insert(pi); });
      for (auto const& pi : providers) {
        auto itinerary =
            route(*w_, *l_, gbfs, e, from, to, m, wheelchair, start_time,
                  std::nullopt, pi, cache, blocked.get(), max);
        if (itinerary.legs_.empty()) {
          continue;
        }
        auto const duration = std::chrono::duration_cast<n::duration_t>(
            std::chrono::seconds{itinerary.duration_});
        if (duration < fastest_direct) {
          fastest_direct = duration;
        }
        itineraries.emplace_back(std::move(itinerary));
      }
    }
  }
  return {itineraries, fastest_direct};
}

void remove_slower_than_fastest_direct(n::routing::query& q) {
  if (!q.fastest_direct_) {
    return;
  }

  constexpr auto const kMaxDuration =
      n::duration_t{std::numeric_limits<n::duration_t::rep>::max()};

  auto const worse_than_fastest_direct = [&](n::duration_t const min) {
    return [&, min](auto const& o) {
      return o.duration() + min >= q.fastest_direct_;
    };
  };
  auto const get_min_duration = [&](auto&& x) {
    return x.empty() ? kMaxDuration
                     : utl::min_element(x, [](auto&& a, auto&& b) {
                         return a.duration() < b.duration();
                       })->duration();
  };

  auto min_start = get_min_duration(q.start_);
  for (auto const& [_, v] : q.td_start_) {
    min_start = std::min(min_start, get_min_duration(v));
  }

  auto min_dest = get_min_duration(q.destination_);
  for (auto const& [_, v] : q.td_dest_) {
    min_dest = std::min(min_dest, get_min_duration(v));
  }

  utl::erase_if(q.start_, worse_than_fastest_direct(min_dest));
  utl::erase_if(q.destination_, worse_than_fastest_direct(min_start));
  for (auto& [k, v] : q.td_start_) {
    utl::erase_if(v, worse_than_fastest_direct(min_dest));
  }
  for (auto& [k, v] : q.td_dest_) {
    utl::erase_if(v, worse_than_fastest_direct(min_start));
  }
}

std::vector<n::routing::via_stop> get_via_stops(
    n::timetable const& tt,
    tag_lookup const& tags,
    std::optional<std::vector<std::string>> const& vias,
    std::vector<std::int64_t> const& times) {
  if (!vias.has_value()) {
    return {};
  }

  auto ret = std::vector<n::routing::via_stop>{};
  for (auto i = 0U; i != vias->size(); ++i) {
    ret.push_back({tags.get_location(tt, (*vias)[i]),
                   n::duration_t{i < times.size() ? times[i] : 0}});
  }
  return ret;
}

api::plan_response routing::operator()(boost::urls::url_view const& url) const {
  auto const rt = rt_;
  auto const rtt = rt->rtt_.get();
  auto const e = rt_->e_.get();
  auto const gbfs = gbfs_;
  if (blocked.get() == nullptr && w_ != nullptr) {
    blocked.reset(new osr::bitvec<osr::node_idx_t>{w_->n_nodes()});
  }

  auto const query = api::plan_params{url.params()};
  auto const deduplicate = [](auto m) {
    utl::erase_duplicates(m);
    return m;
  };
  auto const pre_transit_modes = deduplicate(query.preTransitModes_);
  auto const post_transit_modes = deduplicate(query.postTransitModes_);
  auto const direct_modes = deduplicate(query.directModes_);
  auto const from = get_place(tt_, tags_, query.fromPlace_);
  auto const to = get_place(tt_, tags_, query.toPlace_);
  auto const from_p = to_place(tt_, tags_, w_, pl_, matches_, from);
  auto const to_p = to_place(tt_, tags_, w_, pl_, matches_, to);

  auto const& start = query.arriveBy_ ? to : from;
  auto const& dest = query.arriveBy_ ? from : to;
  auto const& start_modes =
      query.arriveBy_ ? post_transit_modes : pre_transit_modes;
  auto const& dest_modes =
      query.arriveBy_ ? pre_transit_modes : post_transit_modes;

  auto const [start_time, t] = get_start_time(query);

  UTL_START_TIMING(direct);
  auto const [direct, fastest_direct] =
      t.has_value() && !direct_modes.empty() && w_ && l_
          ? route_direct(e, gbfs.get(), from_p, to_p, direct_modes, *t,
                         query.wheelchair_,
                         std::chrono::seconds{query.maxDirectTime_})
          : std::pair{std::vector<api::Itinerary>{}, kInfinityDuration};
  UTL_STOP_TIMING(direct);

  auto odm_task = boost::fibers::packaged_task<
      std::optional<std::vector<n::routing::journey>>()>{[&]() {
    return odm::odm_routing(*this, query, pre_transit_modes, post_transit_modes,
                            direct_modes, from, to, from_p, to_p, start_time);
  }};
  auto odm_journeys = odm_task.get_future();
  boost::fibers::fiber{std::move(odm_task)}.detach();

  if (!query.transitModes_.empty() && fastest_direct > 5min) {
    utl::verify(tt_ != nullptr && tags_ != nullptr,
                "mode=TRANSIT requires timetable to be loaded");

    UTL_START_TIMING(query_preparation);
    auto stats = stats_map_t{};
    auto q = n::routing::query{
        .start_time_ = start_time.start_time_,
        .start_match_mode_ = get_match_mode(start),
        .dest_match_mode_ = get_match_mode(dest),
        .use_start_footpaths_ = !is_intermodal(start),
        .start_ = std::visit(
            utl::overloaded{
                [&](tt_location const l) { return station_start(l.l_); },
                [&](osr::location const& pos) {
                  auto const dir = query.arriveBy_ ? osr::direction::kBackward
                                                   : osr::direction::kForward;
                  return get_offsets(
                      pos, dir, start_modes, query.wheelchair_,
                      std::chrono::seconds{query.maxPreTransitTime_},
                      query.maxMatchingDistance_, gbfs.get(), stats);
                }},
            start),
        .destination_ = std::visit(
            utl::overloaded{
                [&](tt_location const l) { return station_start(l.l_); },
                [&](osr::location const& pos) {
                  auto const dir = query.arriveBy_ ? osr::direction::kForward
                                                   : osr::direction::kBackward;
                  return get_offsets(
                      pos, dir, dest_modes, query.wheelchair_,
                      std::chrono::seconds{query.maxPostTransitTime_},
                      query.maxMatchingDistance_, gbfs.get(), stats);
                }},
            dest),
        .td_start_ =
            rt_->e_ != nullptr
                ? std::visit(
                      utl::overloaded{
                          [&](tt_location) { return td_offsets_t{}; },
                          [&](osr::location const& pos) {
                            auto const dir = query.arriveBy_
                                                 ? osr::direction::kBackward
                                                 : osr::direction::kForward;
                            return get_td_offsets(
                                *e, pos, dir, start_modes, query.wheelchair_,
                                std::chrono::seconds{query.maxPreTransitTime_});
                          }},
                      start)
                : td_offsets_t{},
        .td_dest_ =
            rt_->e_ != nullptr
                ? std::visit(
                      utl::overloaded{
                          [&](tt_location) { return td_offsets_t{}; },
                          [&](osr::location const& pos) {
                            auto const dir = query.arriveBy_
                                                 ? osr::direction::kForward
                                                 : osr::direction::kBackward;
                            return get_td_offsets(
                                *e, pos, dir, dest_modes, query.wheelchair_,
                                std::chrono::seconds{
                                    query.maxPostTransitTime_});
                          }},
                      dest)
                : td_offsets_t{},
        .max_transfers_ = static_cast<std::uint8_t>(
            query.maxTransfers_.has_value() ? *query.maxTransfers_
                                            : n::routing::kMaxTransfers),
        .min_connection_count_ = static_cast<unsigned>(query.numItineraries_),
        .extend_interval_earlier_ = start_time.extend_interval_earlier_,
        .extend_interval_later_ = start_time.extend_interval_later_,
        .prf_idx_ = static_cast<n::profile_idx_t>(query.wheelchair_ ? 2U : 1U),
        .allowed_claszes_ = to_clasz_mask(query.transitModes_),
        .require_bike_transport_ = query.requireBikeTransport_,
        .transfer_time_settings_ =
            n::routing::transfer_time_settings{
                .default_ = (query.minTransferTime_ == 0 &&
                             query.additionalTransferTime_ == 0 &&
                             query.transferTimeFactor_ == 1.0),
                .min_transfer_time_ = n::duration_t{query.minTransferTime_},
                .additional_time_ =
                    n::duration_t{query.additionalTransferTime_},
                .factor_ = static_cast<float>(query.transferTimeFactor_)},
        .via_stops_ =
            get_via_stops(*tt_, *tags_, query.via_, query.viaMinimumStay_),
        .fastest_direct_ = fastest_direct == kInfinityDuration
                               ? std::nullopt
                               : std::optional{fastest_direct}};
    remove_slower_than_fastest_direct(q);
    UTL_STOP_TIMING(query_preparation);

    if (tt_->locations_.footpaths_out_.at(q.prf_idx_).empty()) {
      q.prf_idx_ = 0U;
    }

    if (search_state.get() == nullptr) {
      search_state.reset(new n::routing::search_state{});
    }
    if (raptor_state.get() == nullptr) {
      raptor_state.reset(new n::routing::raptor_state{});
    }

    auto const r = n::routing::raptor_search(
        *tt_, rtt, *search_state, *raptor_state, std::move(q),
        query.arriveBy_ ? n::direction::kBackward : n::direction::kForward,
        std::nullopt);

    odm_journeys.wait();
    if (odm_journeys.valid() && odm_journeys.get().has_value()) {
      // TODO cost-based domination by PT journeys
      // TODO productivity-based domination between remaining ODM journeys
    }

    return {
        .debugOutput_ = join(
            stats,
            stats_map_t{{"direct", UTL_TIMING_MS(direct)},
                        {"query_preparation", UTL_TIMING_MS(query_preparation)},
                        {"n_start_offsets", q.start_.size()},
                        {"n_dest_offsets", q.destination_.size()},
                        {"n_td_start_offsets", q.td_start_.size()},
                        {"n_td_dest_offsets", q.td_dest_.size()}},
            r.search_stats_.to_map(), r.algo_stats_.to_map()),
        .from_ = from_p,
        .to_ = to_p,
        .direct_ = std::move(direct),
        .itineraries_ = utl::to_vec(
            *r.journeys_,
            [&, cache = street_routing_cache_t{}](auto&& j) mutable {
              return journey_to_response(w_, l_, pl_, *tt_, *tags_, e, rtt,
                                         matches_, shapes_, gbfs.get(),
                                         query.wheelchair_, j, start, dest,
                                         cache, blocked.get());
            }),
        .previousPageCursor_ =
            fmt::format("EARLIER|{}", to_seconds(r.interval_.from_)),
        .nextPageCursor_ = fmt::format("LATER|{}", to_seconds(r.interval_.to_)),
    };
  }

  return {.from_ = to_place(tt_, tags_, w_, pl_, matches_, from),
          .to_ = to_place(tt_, tags_, w_, pl_, matches_, to),
          .direct_ = std::move(direct),
          .itineraries_ = {}};
}

}  // namespace motis::ep
