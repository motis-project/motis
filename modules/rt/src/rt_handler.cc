#include "motis/rt/rt_handler.h"

#include "cista/hash.h"

#include "utl/to_vec.h"

#include "utl/pipes.h"
#include "utl/verify.h"

#include "motis/core/common/date_time_util.h"
#include "motis/core/common/logging.h"
#include "motis/core/common/raii.h"
#include "motis/core/schedule/validate_graph.h"
#include "motis/core/conv/trip_conv.h"

#include "motis/module/context/motis_publish.h"
#include "motis/module/message.h"

#include "motis/rt/error.h"
#include "motis/rt/event_resolver.h"
#include "motis/rt/full_trip_handler.h"
#include "motis/rt/reroute.h"
#include "motis/rt/separate_trip.h"
#include "motis/rt/track_change.h"
#include "motis/rt/trip_correction.h"
#include "motis/rt/trip_formation_handler.h"
#include "motis/rt/update_constant_graph.h"
#include "motis/rt/validate_constant_graph.h"
#include "motis/rt/validity_check.h"

using motis::module::msg_ptr;
using namespace motis::logging;
using namespace motis::module;

namespace motis::rt {

rt_handler::rt_handler(schedule& sched, ctx::res_id_t schedule_res_id,
                       bool validate_graph, bool validate_constant_graph,
                       bool print_stats, bool enable_history)
    : sched_(sched),
      schedule_res_id_(schedule_res_id),
      propagator_(sched),
      update_builder_(sched, schedule_res_id),
      msg_history_(enable_history),
      validate_graph_(validate_graph),
      validate_constant_graph_(validate_constant_graph),
      print_stats_(print_stats) {}

msg_ptr rt_handler::update(msg_ptr const& msg) {
  using ris::RISBatch;

  auto const processing_time = now();
  for (auto const& m : *motis_content(RISBatch, msg)->messages()) {
    try {
      update(
          m->message_nested_root(),
          std::string_view{reinterpret_cast<char const*>(m->message()->data()),
                           m->message()->size()},
          processing_time);
    } catch (std::exception const& e) {
      LOG(logging::error) << "rt::on_message: UNEXPECTED ERROR: " << e.what();
    } catch (...) {
      LOG(logging::error) << "rt::on_message: UNEXPECTED UNKNOWN ERROR";
    }
  }
  return nullptr;
}

msg_ptr rt_handler::single(msg_ptr const& msg) {
  using ris::RISMessage;
  update(motis_content(RISMessage, msg), msg->to_string_view(), now());
  return flush(nullptr);
}

void rt_handler::update(motis::ris::RISMessage const* m,
                        std::string_view const msg_buffer,
                        unixtime const processing_time) {
  stats_.count_message(m->content_type());
  count_message(metrics_, m, processing_time);
  auto c = m->content();

  switch (m->content_type()) {
    case ris::RISMessageUnion_DelayMessage: {
      auto const msg = reinterpret_cast<ris::DelayMessage const*>(c);
      stats_.total_updates_ += msg->events()->size();

      auto const reason = (msg->type() == ris::DelayType_Is)
                              ? timestamp_reason::IS
                              : timestamp_reason::FORECAST;

      auto const [trp, resolved] = resolve_events_and_trip(
          stats_, sched_, msg->trip_id(),
          utl::to_vec(*msg->events(),
                      [](ris::UpdatedEvent const* ev) { return ev->base(); }));

      if (trp != nullptr) {
        msg_history_.add(trp->trip_idx_, msg_buffer);
      }

      for (auto i = 0UL; i < resolved.size(); ++i) {
        auto const& resolved_ev = resolved[i];
        if (!resolved_ev) {
          ++stats_.unresolved_events_;
          continue;
        }

        auto const upd_time =
            unix_to_motistime(sched_, msg->events()->Get(i)->updated_time());
        if (upd_time == INVALID_TIME) {
          ++stats_.update_time_out_of_schedule_;
          continue;
        }

        propagator_.add_delay(*resolved_ev, reason, upd_time);
        ++stats_.found_updates_;
      }

      break;
    }

    case ris::RISMessageUnion_AdditionMessage: {
      auto const [result, trp] =
          additional_service_builder(stats_, sched_, update_builder_)
              .build_additional_train(
                  reinterpret_cast<ris::AdditionMessage const*>(c));
      stats_.count_additional(result);
      if (trp != nullptr) {
        msg_history_.add(trp->trip_idx_, msg_buffer);
      }
      break;
    }

    case ris::RISMessageUnion_CancelMessage: {
      auto const msg = reinterpret_cast<ris::CancelMessage const*>(c);

      propagate();

      std::vector<ev_key> cancelled_evs;
      auto const result = reroute(
          stats_, sched_, cancelled_delays_, cancelled_evs, msg->trip_id(),
          utl::to_vec(*msg->events()), {}, update_builder_);

      if (result.first == reroute_result::OK) {
        for (auto const& e : *result.second->edges_) {
          propagator_.recalculate(ev_key{e, 0, event_type::DEP});
          propagator_.recalculate(ev_key{e, 0, event_type::ARR});
        }
        for (auto const& e : cancelled_evs) {
          propagator_.recalculate(e);
        }
      }

      break;
    }

    case ris::RISMessageUnion_RerouteMessage: {
      auto const msg = reinterpret_cast<ris::RerouteMessage const*>(c);

      propagate();

      std::vector<ev_key> cancelled_evs;
      auto const [result, trp] =
          reroute(stats_, sched_, cancelled_delays_, cancelled_evs,
                  msg->trip_id(), utl::to_vec(*msg->cancelled_events()),
                  utl::to_vec(*msg->new_events()), update_builder_);

      stats_.count_reroute(result);

      if (result == reroute_result::OK) {
        for (auto const& e : *trp->edges_) {
          propagator_.recalculate(ev_key{e, 0, event_type::DEP});
          propagator_.recalculate(ev_key{e, 0, event_type::ARR});
        }
        for (auto const& e : cancelled_evs) {
          propagator_.recalculate(e);
        }
        msg_history_.add(trp->trip_idx_, msg_buffer);
      }

      break;
    }

    case ris::RISMessageUnion_TrackMessage: {
      auto const msg = reinterpret_cast<ris::TrackMessage const*>(c);

      stats_.total_evs_ += msg->events()->size();

      auto const resolve = [&]() {
        return resolve_events_and_trip(
            stats_, sched_, msg->trip_id(),
            utl::to_vec(*msg->events(), [](ris::UpdatedTrack const* ev) {
              return ev->base();
            }));
      };
      auto [trp, resolved] = resolve();
      auto new_tracks = std::vector<uint16_t>(msg->events()->size());

      trip* separate_trp = nullptr;
      for (auto i = 0UL; i < resolved.size(); ++i) {
        auto const& k = resolved[i];
        if (!k) {
          continue;
        }

        auto const ev = msg->events()->Get(i);
        auto const new_track = static_cast<uint16_t>(
            get_track(sched_, ev->updated_track()->str()));
        new_tracks[i] = new_track;

        if (separate_trp == nullptr && !fits_edge(sched_, *k, new_track)) {
          separate_trp = sched_.merged_trips_[k->lcon()->trips_]->front();
        }
      }

      if (separate_trp != nullptr) {
        separate_trip(sched_, separate_trp, update_builder_);
        auto r = resolve();
        trp = r.first;
        resolved = std::move(r.second);
        stats_.track_separations_++;
      }

      for (auto i = 0UL; i < resolved.size(); ++i) {
        auto const& k = resolved[i];
        if (!k) {
          continue;
        }

        auto const ev = msg->events()->Get(i);
        auto const new_track = new_tracks[i];

        if (auto const it = sched_.graph_to_schedule_track_index_.find(*k);
            it == sched_.graph_to_schedule_track_index_.end()) {
          sched_.graph_to_schedule_track_index_[*k] =
              k->ev_type_ == event_type::ARR ? k->lcon()->full_con_->a_track_
                                             : k->lcon()->full_con_->d_track_;
        }

        track_events_.emplace_back(
            track_info{*k, ev->updated_track()->str(),
                       unix_to_motistime(sched_, ev->base()->schedule_time())});

        update_track(sched_, *k, new_track);
      }

      if (trp != nullptr) {
        msg_history_.add(trp->trip_idx_, msg_buffer);
      }
      break;
    }

    case ris::RISMessageUnion_FreeTextMessage: {
      auto const msg = reinterpret_cast<ris::FreeTextMessage const*>(c);
      stats_.total_evs_ += msg->events()->size();
      auto const [trp, resolved] = resolve_events_and_trip(
          stats_, sched_, msg->trip_id(),
          utl::to_vec(*msg->events(), [](ris::Event const* ev) { return ev; }));
      if (trp == nullptr) {
        return;
      }
      auto const events = utl::all(resolved)  //
                          | utl::remove_if([](auto&& k) { return !k; })  //
                          | utl::transform([](auto&& k) { return *k; })  //
                          | utl::vec();
      auto const ft =
          free_text{msg->free_text()->code(), msg->free_text()->text()->str(),
                    msg->free_text()->type()->str()};
      for (auto const& k : events) {
        sched_.graph_to_free_texts_[k].emplace(ft);
      }
      free_text_events_.emplace_back(free_texts{trp, ft, events});
      msg_history_.add(trp->trip_idx_, msg_buffer);
      break;
    }

    case ris::RISMessageUnion_FullTripMessage: {
      handle_full_trip_msg(
          stats_, sched_, update_builder_, msg_history_, propagator_,
          reinterpret_cast<ris::FullTripMessage const*>(c), msg_buffer,
          cancelled_delays_, metrics_, m->timestamp(), processing_time);
      break;
    }

    case ris::RISMessageUnion_TripFormationMessage: {
      handle_trip_formation_msg(
          stats_, sched_, update_builder_,
          reinterpret_cast<ris::TripFormationMessage const*>(c), metrics_,
          m->timestamp(), processing_time);
      break;
    }

    default: break;
  }

  batch_updates();
}

void rt_handler::propagate() {
  MOTIS_FINALLY([this]() { propagator_.reset(); });

  propagator_.propagate();

  std::set<trip const*> trips_to_correct;
  std::set<trip::route_edge> updated_route_edges;
  for (auto const& di : propagator_.events()) {
    auto const& k = di->get_ev_key();
    auto const t = di->get_current_time();

    if (k.is_canceled()) {
      continue;
    }

    auto const edge_fit = fits_edge(k, t);
    auto const trip_fit = fits_trip(sched_, k, t);

    stats_.edge_fit_ += !edge_fit ? 1 : 0;
    stats_.trip_fit_ += !trip_fit ? 1 : 0;
    stats_.edge_fit_and_trip_fit_ += (!edge_fit && !trip_fit) ? 1 : 0;
    ++stats_.total_;

    if (!edge_fit || !trip_fit) {
      stats_.edge_fit_or_trip_fit_ += (!edge_fit || !trip_fit) ? 1 : 0;
      auto const trp = sched_.merged_trips_[k.lcon()->trips_]->front();

      if (!edge_fit) {
        separate_trip(sched_, trp, update_builder_);
      }

      if (!trip_fit) {
        trips_to_correct.insert(trp);
      }
    }

    auto const& updated_k = di->get_ev_key();
    if (!updated_k.is_canceled()) {
      auto& event_time = updated_k.ev_type_ == event_type::DEP
                             ? updated_k.lcon()->d_time_
                             : updated_k.lcon()->a_time_;
      const_cast<time&>(event_time) = t;  // NOLINT
      updated_route_edges.insert(updated_k.route_edge_);
    }

    update_builder_.add_delay(di);
    batch_updates();
  }

  for (auto const& trp : trips_to_correct) {
    for (auto const& di : trip_corrector(sched_, trp).fix_times()) {
      update_builder_.add_delay(di);
      updated_route_edges.insert(di->get_ev_key().route_edge_);

      if (!fits_edge(di->get_ev_key(), di->get_current_time())) {
        ++stats_.edge_fit_1_;
        separate_trip(sched_, di->get_ev_key(), update_builder_);
      }
      batch_updates();
    }
  }

  for (auto const& re : updated_route_edges) {
    constant_graph_add_route_edge(sched_, re);
  }

  stats_.propagated_updates_ = propagator_.events().size();
  stats_.graph_updates_ = update_builder_.delay_count();

  // tracks
  for (auto const& t : track_events_) {
    update_builder_.add_track_nodes(t.event_, t.track_, t.schedule_time_);
  }

  // free_texts
  for (auto const& f : free_text_events_) {
    update_builder_.add_free_text_nodes(f.trp_, f.ft_, f.events_);
  }

  ctx::await_all(motis_publish(update_builder_.finish()));

  update_builder_.reset();
  track_events_.clear();
  free_text_events_.clear();
}

void rt_handler::batch_updates() {
  if (update_builder_.should_finish()) {
    ctx::await_all(motis_publish(update_builder_.finish()));
    update_builder_.reset();
  }
}

msg_ptr rt_handler::flush(msg_ptr const&) {
  scoped_timer const t("flush");

  MOTIS_FINALLY([this]() {
    if (print_stats_) {
      stats_.print();
    }
    stats_ = statistics();
    propagator_.reset();
    update_builder_.reset();
    track_events_.clear();
    free_text_events_.clear();
  });

  propagate();

  if (validate_graph_) {
    validate_graph(sched_);
  }
  if (validate_constant_graph_) {
    validate_constant_graph(sched_);
  }

  motis::module::message_creator mc;
  mc.create_and_finish(MsgContent_RtGraphUpdated,
                       CreateRtGraphUpdated(mc, schedule_res_id_).Union(),
                       "/rt/graph_updated");
  ctx::await_all(motis_publish(module::make_msg(mc)));

  if (stats_.sanity_check_fails()) {
    return motis::module::make_error_msg(error::sanity_check_failed);
  } else {
    return nullptr;
  }
}

}  // namespace motis::rt
