#include "motis/rt/rt_handler.h"

#include "utl/to_vec.h"

#include "utl/pipes.h"
#include "utl/verify.h"

#include "motis/core/common/logging.h"
#include "motis/core/common/raii.h"
#include "motis/core/conv/trip_conv.h"

#include "motis/module/context/get_schedule.h"
#include "motis/module/context/motis_publish.h"

#include "motis/rt/event_resolver.h"
#include "motis/rt/reroute.h"
#include "motis/rt/separate_trip.h"
#include "motis/rt/shifted_nodes_msg_builder.h"
#include "motis/rt/trip_correction.h"
#include "motis/rt/update_constant_graph.h"
#include "motis/rt/validate_constant_graph.h"
#include "motis/rt/validate_graph.h"
#include "motis/rt/validity_check.h"

using motis::module::msg_ptr;
using namespace motis::logging;

namespace motis::rt {

namespace {

void add_free_text_nodes(flatbuffers::FlatBufferBuilder& fbb,
                         schedule const& sched, trip const* trp,
                         free_text const& ft, std::vector<ev_key> const& events,
                         std::vector<flatbuffers::Offset<RtUpdate>>& updates) {
  auto const trip = to_fbs(sched, fbb, trp);
  auto const r = Range{0, 0};
  auto const free_text =
      CreateFreeText(fbb, &r, ft.code_, fbb.CreateString(ft.text_),
                     fbb.CreateString(ft.type_));
  for (auto const& k : events) {
    updates.emplace_back(CreateRtUpdate(
        fbb, Content_RtFreeTextUpdate,
        CreateRtFreeTextUpdate(
            fbb,
            CreateRtEventInfo(
                fbb, trip,
                fbb.CreateString(
                    sched.stations_.at(k.get_station_idx())->eva_nr_),
                motis_to_unixtime(
                    sched, k ? get_schedule_time(sched, k) : INVALID_TIME),
                to_fbs(k.ev_type_)),
            free_text)
            .Union()));
  }
}

void add_track_nodes(flatbuffers::FlatBufferBuilder& fbb, schedule const& sched,
                     ev_key const& k, std::string const& track,
                     motis::time const schedule_time,
                     std::vector<flatbuffers::Offset<RtUpdate>>& updates) {
  auto const trip =
      to_fbs(sched, fbb, sched.merged_trips_[k.lcon()->trips_]->at(0));
  updates.emplace_back(CreateRtUpdate(
      fbb, Content_RtTrackUpdate,
      CreateRtTrackUpdate(
          fbb,
          CreateRtEventInfo(
              fbb, trip,
              fbb.CreateString(
                  sched.stations_.at(k.get_station_idx())->eva_nr_),
              motis_to_unixtime(sched, schedule_time), to_fbs(k.ev_type_)),
          fbb.CreateString(track))
          .Union()));
}

}  // namespace

rt_handler::rt_handler(schedule& sched, bool validate_graph,
                       bool validate_constant_graph)
    : sched_(sched),
      propagator_(sched),
      validate_graph_(validate_graph),
      validate_constant_graph_(validate_constant_graph) {}

msg_ptr rt_handler::update(msg_ptr const& msg) {
  using ris::RISBatch;

  auto& s = module::get_schedule();
  for (auto const& m : *motis_content(RISBatch, msg)->messages()) {
    update(s, m->message_nested_root());
  }
  return nullptr;
}

msg_ptr rt_handler::single(msg_ptr const& msg) {
  using ris::Message;
  update(module::get_schedule(), motis_content(Message, msg));
  flush(nullptr);
  return nullptr;
}

void rt_handler::update(schedule& s, motis::ris::Message const* m) {
  stats_.count_message(nested->content_type());
  auto c = nested->content();
  try {
    switch (nested->content_type()) {
      case ris::MessageUnion_DelayMessage: {
        auto const msg = reinterpret_cast<ris::DelayMessage const*>(c);
        stats_.total_updates_ += msg->events()->size();

        auto const reason = (msg->type() == ris::DelayType_Is)
                                ? timestamp_reason::IS
                                : timestamp_reason::FORECAST;

        auto const resolved = resolve_events(
            stats_, s, msg->trip_id(),
            utl::to_vec(*msg->events(), [](ris::UpdatedEvent const* ev) {
              return ev->base();
            }));

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
        auto result = additional_service_builder(s).build_additional_train(
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
                    utl::to_vec(*msg->events()), {});

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
                    utl::to_vec(*msg->new_events()));

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
            utl::to_vec(*msg->events(), [](ris::UpdatedTrack const* ev) {
              return ev->base();
            }));

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

          track_events_.emplace_back(track_info{
              *k, ev->updated_track()->str(),
              unix_to_motistime(sched_, ev->base()->schedule_time())});

          auto fcon = *k->lcon()->full_con_;
          (k->ev_type_ == event_type::ARR ? fcon.a_track_ : fcon.d_track_) =
              get_track(s, ev->updated_track()->str());

          const_cast<light_connection*>(k->lcon())->full_con_ =  // NOLINT
              s.full_connections_
                  .emplace_back(mcd::make_unique<connection>(fcon))
                  .get();
        }
        break;
      }

      case ris::MessageUnion_FreeTextMessage: {
        auto const msg = reinterpret_cast<ris::FreeTextMessage const*>(c);
        stats_.total_evs_ += msg->events()->size();
        auto const [trp, resolved] = resolve_events_and_trip(
            stats_, s, msg->trip_id(),
            utl::to_vec(*msg->events(),
                        [](ris::Event const* ev) { return ev; }));
        if (trp == nullptr) {
          continue;
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
  } catch (std::exception const& e) {
    printf("rt::on_message: UNEXPECTED ERROR: %s\n", e.what());
  } catch (...) {
  }
}

msg_ptr get_rt_updates(motis::module::message_creator& fbb,
                       std::vector<flatbuffers::Offset<RtUpdate>>& updates) {
  fbb.create_and_finish(MsgContent_RtUpdates,
                        CreateRtUpdates(fbb, fbb.CreateVector(updates)).Union(),
                        "/rt/update", DestinationType_Topic);
  return make_msg(fbb);
}

void rt_handler::propagate() {
  MOTIS_FINALLY([this]() { propagator_.reset(); });

  propagator_.propagate();

  std::set<trip const*> trips_to_correct;
  std::set<trip::route_edge> updated_route_edges;
  motis::module::message_creator fbb;
  std::vector<flatbuffers::Offset<RtUpdate>> updates;
  shifted_nodes_msg_builder shifted_nodes(fbb, sched_);
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

    shifted_nodes.add(di);
  }

  for (auto const& trp : trips_to_correct) {
    assert(trp->lcon_idx_ == 0 &&
           trp->edges_->front()->m_.route_edge_.conns_.size() == 1);
    for (auto const& di : trip_corrector(sched_, trp).fix_times()) {
      shifted_nodes.add(di);
      updated_route_edges.insert(di->get_ev_key().route_edge_);
    }
  }

  for (auto const& re : updated_route_edges) {
    constant_graph_add_route_edge(sched_, re);
  }

  stats_.propagated_updates_ = propagator_.events().size();
  stats_.graph_updates_ = shifted_nodes.size();

  // delays
  shifted_nodes.finish(updates);

  // tracks
  for (auto const& t : track_events_) {
    add_track_nodes(fbb, sched_, t.event_, t.track_, t.schedule_time_, updates);
  }

  // free_texts
  for (auto const& f : free_text_events_) {
    add_free_text_nodes(fbb, sched_, f.trp_, f.ft_, f.events_, updates);
  }

  ctx::await_all(motis_publish(get_rt_updates(fbb, updates)));

  fbb.Clear();
  updates.clear();
}

msg_ptr rt_handler::flush(msg_ptr const&) {
  scoped_timer t("flush");

  MOTIS_FINALLY([this]() {
    stats_.print();
    stats_ = statistics();
    propagator_.reset();
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

  return nullptr;
}

}  // namespace motis::rt
