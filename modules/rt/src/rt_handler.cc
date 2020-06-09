#include "motis/rt/rt_handler.h"

#include "utl/to_vec.h"

#include "utl/pipes.h"
#include "utl/verify.h"

#include "motis/core/common/logging.h"
#include "motis/core/common/raii.h"
#include "motis/core/conv/trip_conv.h"

#include "motis/module/context/get_schedule.h"
#include "motis/module/context/motis_publish.h"

#include "motis/rt/error.h"
#include "motis/rt/event_resolver.h"
#include "motis/rt/reroute.h"
#include "motis/rt/separate_trip.h"
#include "motis/rt/trip_correction.h"
#include "motis/rt/update_constant_graph.h"
#include "motis/rt/validate_constant_graph.h"
#include "motis/rt/validate_graph.h"
#include "motis/rt/validity_check.h"

using motis::module::msg_ptr;
using namespace motis::logging;

namespace motis::rt {

rt_handler::rt_handler(schedule& sched, bool validate_graph,
                       bool validate_constant_graph)
    : sched_(sched),
      propagator_(sched),
      update_builder_(sched),
      validate_graph_(validate_graph),
      validate_constant_graph_(validate_constant_graph) {}

msg_ptr rt_handler::update(msg_ptr const& msg) {
  using ris::RISBatch;

  auto& s = module::get_schedule();
  for (auto const& m : *motis_content(RISBatch, msg)->messages()) {
    try {
      update(s, m->message_nested_root());
    } catch (std::exception const& e) {
      printf("rt::on_message: UNEXPECTED ERROR: %s\n", e.what());
    } catch (...) {
      printf("rt::on_message: UNEXPECTED UNKNOWN ERROR\n");
    }
  }
  return nullptr;
}

msg_ptr rt_handler::single(msg_ptr const& msg) {
  using ris::Message;
  update(module::get_schedule(), motis_content(Message, msg));
  return flush(nullptr);
}

void rt_handler::update(schedule& s, motis::ris::Message const* m) {
  stats_.count_message(m->content_type());
  auto c = m->content();

  switch (m->content_type()) {
    case ris::MessageUnion_DelayMessage: {
      auto const msg = reinterpret_cast<ris::DelayMessage const*>(c);
      stats_.total_updates_ += msg->events()->size();

      auto const reason = (msg->type() == ris::DelayType_Is)
                              ? timestamp_reason::IS
                              : timestamp_reason::FORECAST;

      auto const resolved = resolve_events(
          stats_, s, msg->trip_id(),
          utl::to_vec(*msg->events(),
                      [](ris::UpdatedEvent const* ev) { return ev->base(); }));

      for (auto i = 0UL; i < resolved.size(); ++i) {
        auto const& resolved_ev = resolved[i];
        if (!resolved_ev) {
          ++stats_.unresolved_events_;
          continue;
        }

        auto const upd_time =
            unix_to_motistime(s, msg->events()->Get(i)->updated_time());
        if (upd_time == INVALID_TIME) {
          ++stats_.update_time_out_of_schedule_;
          continue;
        }

        propagator_.add_delay(*resolved_ev, reason, upd_time);
        ++stats_.found_updates_;
      }

      break;
    }

    case ris::MessageUnion_AdditionMessage: {
      auto result = additional_service_builder(s, update_builder_)
                        .build_additional_train(
                            reinterpret_cast<ris::AdditionMessage const*>(c));
      stats_.count_additional(result);
      break;
    }

    case ris::MessageUnion_CancelMessage: {
      auto const msg = reinterpret_cast<ris::CancelMessage const*>(c);

      propagate();

      std::vector<ev_key> cancelled_evs;
      auto const result =
          reroute(stats_, s, cancelled_delays_, cancelled_evs, msg->trip_id(),
                  utl::to_vec(*msg->events()), {}, update_builder_);

      if (result.first == reroute_result::OK) {
        for (auto const& e : *result.second->edges_) {
          propagator_.add_delay(ev_key{e, 0, event_type::DEP});
          propagator_.add_delay(ev_key{e, 0, event_type::ARR});
        }
        for (auto const& e : cancelled_evs) {
          propagator_.add_canceled(e);
        }
      }

      break;
    }

    case ris::MessageUnion_RerouteMessage: {
      auto const msg = reinterpret_cast<ris::RerouteMessage const*>(c);

      propagate();

      std::vector<ev_key> cancelled_evs;
      auto const result =
          reroute(stats_, s, cancelled_delays_, cancelled_evs, msg->trip_id(),
                  utl::to_vec(*msg->cancelled_events()),
                  utl::to_vec(*msg->new_events()), update_builder_);

      stats_.count_reroute(result.first);

      if (result.first == reroute_result::OK) {
        for (auto const& e : *result.second->edges_) {
          propagator_.add_delay(ev_key{e, 0, event_type::DEP});
          propagator_.add_delay(ev_key{e, 0, event_type::ARR});
        }
        for (auto const& e : cancelled_evs) {
          propagator_.add_canceled(e);
        }
      }

      break;
    }

    case ris::MessageUnion_TrackMessage: {
      auto const msg = reinterpret_cast<ris::TrackMessage const*>(c);

      stats_.total_evs_ += msg->events()->size();

      auto const resolved = resolve_events(
          stats_, s, msg->trip_id(),
          utl::to_vec(*msg->events(),
                      [](ris::UpdatedTrack const* ev) { return ev->base(); }));

      for (auto i = 0UL; i < resolved.size(); ++i) {
        auto const& k = resolved[i];
        if (!k) {
          continue;
        }

        if (auto const it = s.graph_to_track_index_.find(*k);
            it == s.graph_to_track_index_.end()) {
          s.graph_to_track_index_[*k] = k->ev_type_ == event_type::ARR
                                            ? k->lcon()->full_con_->a_track_
                                            : k->lcon()->full_con_->d_track_;
        }

        auto const ev = msg->events()->Get(i);

        track_events_.emplace_back(
            track_info{*k, ev->updated_track()->str(),
                       unix_to_motistime(sched_, ev->base()->schedule_time())});

        auto fcon = *k->lcon()->full_con_;
        (k->ev_type_ == event_type::ARR ? fcon.a_track_ : fcon.d_track_) =
            get_track(s, ev->updated_track()->str());

        const_cast<light_connection*>(k->lcon())->full_con_ =  // NOLINT
            s.full_connections_.emplace_back(mcd::make_unique<connection>(fcon))
                .get();
      }
      break;
    }

    case ris::MessageUnion_FreeTextMessage: {
      auto const msg = reinterpret_cast<ris::FreeTextMessage const*>(c);
      stats_.total_evs_ += msg->events()->size();
      auto const [trp, resolved] = resolve_events_and_trip(
          stats_, s, msg->trip_id(),
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
        s.graph_to_free_texts_[k].emplace(ft);
      }
      free_text_events_.emplace_back(free_texts{trp, ft, events});
      break;
    }

    default: break;
  }
}

void rt_handler::propagate() {
  MOTIS_FINALLY([this]() { propagator_.reset(); });

  propagator_.propagate();

  std::set<trip const*> trips_to_correct;
  std::set<trip::route_edge> updated_route_edges;
  for (auto const& di : propagator_.events()) {
    auto const& k = di->get_ev_key();
    auto const t = di->get_current_time();

    auto const edge_fit = fits_edge(k, t);
    auto const trip_fit = fits_trip(sched_, k, t);
    if (!edge_fit || !trip_fit) {
      auto const trp = sched_.merged_trips_[k.lcon()->trips_]->front();
      seperate_trip(sched_, trp);

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
  }

  for (auto const& trp : trips_to_correct) {
    assert(trp->lcon_idx_ == 0 &&
           trp->edges_->front()->m_.route_edge_.conns_.size() == 1);
    for (auto const& di : trip_corrector(sched_, trp).fix_times()) {
      update_builder_.add_delay(di);
      updated_route_edges.insert(di->get_ev_key().route_edge_);
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
}

msg_ptr rt_handler::flush(msg_ptr const&) {
  scoped_timer t("flush");

  MOTIS_FINALLY([this]() {
    stats_.print();
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

  if (stats_.sanity_check_fails()) {
    return motis::module::make_error_msg(error::sanity_check_failed);
  } else {
    return nullptr;
  }
}

}  // namespace motis::rt
