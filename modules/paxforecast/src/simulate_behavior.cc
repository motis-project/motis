#include "motis/paxforecast/simulate_behavior.h"

#include "utl/zip.h"

#include "motis/core/debug/trip.h"

#include "motis/paxmon/compact_journey_util.h"
#include "motis/paxmon/messages.h"
#include "motis/paxmon/reachability.h"
#include "motis/paxmon/temp_passenger_group.h"

#include "motis/paxforecast/behavior/util.h"

using namespace motis::paxmon;
using namespace flatbuffers;

namespace motis::paxforecast {

void merge_journeys_failed(schedule const& sched, universe& uv,
                           affected_route_info const& ar,
                           passenger_localization const& loc,
                           alternative const& alt,
                           fws_compact_journey const& old_journey,
                           compact_journey const& journey_prefix) {
  std::cout << "simulate_behavior_for_route: merge_journeys failed for group "
            << ar.pgwrap_.pgwr_.pg_ << "." << ar.pgwrap_.pgwr_.route_ << "\n";
  std::cout << "\noriginal planned journey:\n";
  print_compact_journey(sched, old_journey);

  auto const print_localization = [&](passenger_localization const& loc) {
    std::cout << "localization: in_trip=" << loc.in_trip()
              << ", first_station=" << loc.first_station_
              << ", station=" << loc.at_station_->name_.str()
              << ", schedule_arrival_time="
              << format_time(loc.schedule_arrival_time_)
              << ", current_arrival_time="
              << format_time(loc.current_arrival_time_) << "\n";
    if (loc.in_trip()) {
      std::cout << "in trip:\n";
      print_trip(sched, loc.in_trip_);
    }
  };

  print_localization(loc);

  std::cout << "\ntrying to merge journeys:\nprefix:\n";
  print_compact_journey(sched, journey_prefix);
  std::cout << "\nsuffix:\n";
  print_compact_journey(sched, alt.compact_journey_);

  std::cout << "\ntrips:" << std::endl;
  for (auto const& leg : old_journey.legs()) {
    std::cout << motis::debug::trip_with_sections{sched,
                                                  get_trip(sched,
                                                           leg.trip_idx_)}
              << std::endl;
  }

  auto const current_time =
      unix_to_motistime(sched.schedule_begin_, sched.system_time_);
  auto const search_time =
      static_cast<time>(current_time + uv.preparation_time_);

  auto const reachability = get_reachability(uv, old_journey);
  std::cout << "reachability: ok=" << reachability.ok_
            << ", status=" << reachability.status_ << "\n";
  for (auto const& rt : reachability.reachable_trips_) {
    std::cout << "  trip: "
              << motis::debug::trip{sched, get_trip(sched, rt.trip_idx_)}
              << "\n"
              << "    enter: sched=" << format_time(rt.enter_schedule_time_)
              << ", real=" << format_time(rt.enter_real_time_)
              << ", edge_idx=" << rt.enter_edge_idx_
              << "\n    exit: sched=" << format_time(rt.exit_schedule_time_)
              << ", real=" << format_time(rt.exit_real_time_)
              << ", edge_idx=" << rt.exit_edge_idx_ << "\n";
  }
  for (auto const& rs : reachability.reachable_interchange_stations_) {
    std::cout << "  station: " << sched.stations_.at(rs.station_)->name_
              << ", sched=" << format_time(rs.schedule_time_)
              << ", real=" << format_time(rs.real_time_) << std::endl;
  }
  if (reachability.first_unreachable_transfer_) {
    auto const& bt = *reachability.first_unreachable_transfer_;
    std::cout << "  first unreachable transfer: leg=" << bt.leg_index_
              << ", direction="
              << (bt.direction_ == transfer_direction_t::ENTER ? "ENTER"
                                                               : "EXIT")
              << ", curr_arr=" << format_time(bt.current_arrival_time_)
              << ", curr_dep=" << format_time(bt.current_departure_time_)
              << ", transfer_time=" << bt.required_transfer_time_
              << ", arr_canceled=" << bt.arrival_canceled_
              << ", dep_canceled=" << bt.departure_canceled_ << std::endl;
  }

  auto const loc_now = localize(sched, reachability, current_time);
  auto const loc_prep = localize(sched, reachability, search_time);

  std::cout << "\nlocalization @current_time " << format_time(current_time)
            << ":\n";
  print_localization(loc_now);
  std::cout << "\nlocalization @search_time " << format_time(search_time)
            << ":\n";
  print_localization(loc_prep);
  std::cout << std::endl;
}

void simulate_behavior_for_route(
    schedule const& sched, universe& uv, update_groups_context& ug_ctx,
    simulation_options const& options, affected_route_info const& ar,
    std::vector<alternative> const& alts_now,
    std::vector<alternative> const& alts_broken,
    reroute_reason_t const default_reroute_reason) {
  auto const use_uninformed_pax = options.uninformed_pax_ != 0.F;
  auto const use_stay =
      use_uninformed_pax &&
      default_reroute_reason == reroute_reason_t::MAJOR_DELAY_EXPECTED;
  auto& gr = uv.passenger_groups_.route(ar.pgwrap_.pgwr_);

  auto new_routes = std::vector<Offset<PaxMonGroupRoute>>{};

  auto const old_journey =
      uv.passenger_groups_.journey(gr.compact_journey_index_);
  auto rerouted_to_old_journey = false;

  auto const process_alts = [&](std::vector<alternative> const& alts,
                                passenger_localization const& loc,
                                float const base_prob) {
    auto const journey_prefix = get_prefix(sched, old_journey, loc);
    auto const probs = behavior::calc_new_probabilites(
        base_prob, alts, options.probability_threshold_);

    for (auto const& [alt, new_prob] : utl::zip(alts, probs)) {
      if (new_prob == 0.F) {
        continue;
      }

      auto new_journey = compact_journey{};
      try {
        new_journey =
            merge_journeys(sched, journey_prefix, alt.compact_journey_);
      } catch (std::runtime_error const& e) {
        merge_journeys_failed(sched, uv, ar, loc, alt, old_journey,
                              journey_prefix);
        throw e;
      }

      auto prob = new_prob;
      if (use_stay && new_journey == old_journey) {
        prob += ar.pgwrap_.probability_ * options.uninformed_pax_;
        rerouted_to_old_journey = true;
      }

      new_routes.emplace_back(
          to_fbs(sched, ug_ctx.mc_,
                 temp_group_route{
                     .index_ = std::nullopt,
                     .probability_ = prob,
                     .journey_ = new_journey,
                     .planned_arrival_time_ = gr.planned_arrival_time_,
                     .estimated_delay_ = 0 /* updated by reroute groups api */,
                     .source_flags_ = route_source_flags::FORECAST,
                     .planned_ = false}));
    }
  };

  if (!alts_now.empty()) {
    auto base_prob = ar.pgwrap_.probability_;
    if (use_uninformed_pax && (!alts_broken.empty() || use_stay)) {
      base_prob *= (1.F - options.uninformed_pax_);
    }
    process_alts(alts_now, ar.loc_now_, base_prob);
    if (use_stay && !rerouted_to_old_journey) {
      new_routes.emplace_back(to_fbs(
          sched, ug_ctx.mc_,
          temp_group_route{
              .index_ = ar.pgwrap_.pgwr_.route_,
              .probability_ = ar.pgwrap_.probability_ * options.uninformed_pax_,
              .journey_ = {},
              .planned_arrival_time_ = gr.planned_arrival_time_,
              .estimated_delay_ = 0 /* updated by reroute groups api */,
              .source_flags_ = route_source_flags::FORECAST,
              .planned_ = false}));
      rerouted_to_old_journey = true;
    }
  }

  if (!alts_broken.empty() && use_uninformed_pax) {
    auto const base_prob =
        alts_now.empty() ? ar.pgwrap_.probability_
                         : ar.pgwrap_.probability_ * options.uninformed_pax_;
    process_alts(alts_broken, ar.loc_broken_, base_prob);
  }

  if (new_routes.size() == 1 && rerouted_to_old_journey) {
    return;
  }

  auto reroute_reason = default_reroute_reason;
  if (alts_now.empty() && alts_broken.empty()) {
    if (default_reroute_reason == reroute_reason_t::MAJOR_DELAY_EXPECTED) {
      return;
    }
    // keep existing group (only reachable part)
    reroute_reason = reroute_reason_t::DESTINATION_UNREACHABLE;
    gr.destination_unreachable_ = true;
  }

  ug_ctx.reroutes_.emplace_back(CreatePaxMonRerouteGroup(
      ug_ctx.mc_, ar.pgwrap_.pgwr_.pg_, ar.pgwrap_.pgwr_.route_,
      ug_ctx.mc_.CreateVector(new_routes),
      static_cast<PaxMonRerouteReason>(reroute_reason),
      broken_transfer_info_to_fbs(ug_ctx.mc_, sched, ar.broken_transfer_info_),
      false,
      ug_ctx.mc_.CreateVector(std::vector<Offset<PaxMonLocalizationWrapper>>{
          to_fbs_localization_wrapper(sched, ug_ctx.mc_, ar.loc_now_)})));
}

}  // namespace motis::paxforecast
