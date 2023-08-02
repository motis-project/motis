#include "motis/rt/full_trip_handler.h"

#include <cassert>
#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "boost/uuid/nil_generator.hpp"
#include "boost/uuid/uuid_io.hpp"

#include "utl/enumerate.h"
#include "utl/erase.h"
#include "utl/to_vec.h"
#include "utl/verify.h"
#include "utl/zip.h"

#include "motis/core/common/logging.h"
#include "motis/core/schedule/timestamp_reason.h"
#include "motis/core/access/realtime_access.h"
#include "motis/core/access/station_access.h"
#include "motis/core/access/trip_iterator.h"
#include "motis/core/access/uuids.h"
#include "motis/core/conv/trip_conv.h"
#include "motis/core/debug/trip.h"

#include "motis/rt/build_route_node.h"
#include "motis/rt/connection_builder.h"
#include "motis/rt/expanded_trips.h"
#include "motis/rt/incoming_edges.h"
#include "motis/rt/update_constant_graph.h"
#include "motis/rt/util.h"

using namespace motis::logging;
using namespace motis::ris;

namespace motis::rt {

inline timestamp_reason from_fbs(TimestampType const t) {
  switch (t) {
    case TimestampType_Schedule: return timestamp_reason::SCHEDULE;
    case TimestampType_Is: return timestamp_reason::IS;
    case TimestampType_Unknown:  // TODO(pablo): ?
    case TimestampType_Forecast: return timestamp_reason::FORECAST;
    default: return timestamp_reason::SCHEDULE;
  }
}

namespace {

struct event_info {
  station_node* station_{};
  bool interchange_allowed_{};
  time schedule_time_{INVALID_TIME};
  time current_time_{INVALID_TIME};
  timestamp_reason timestamp_reason_{timestamp_reason::SCHEDULE};
  std::uint16_t schedule_track_{};
  std::uint16_t current_track_{};
  ev_key ev_key_;
  boost::uuids::uuid uuid_{};

  inline bool time_updated(event_info const& o) const {
    return current_time_ != o.current_time_ ||
           timestamp_reason_ != o.timestamp_reason_;
  }

  inline bool track_updated(event_info const& o) const {
    return current_track_ != o.current_track_;
  }

  inline ev_key const& get_ev_key() const {
    assert(ev_key_.is_not_null());
    return ev_key_;
  }
};

struct section {
  light_connection* lcon() const {
    assert(lc_ != nullptr);
    return lc_;
  }

  event_info dep_;
  event_info arr_;
  light_connection* lc_{};
};

struct trip_backup {
  mcd::vector<trip::route_edge> edges_;
  lcon_idx_t lcon_idx_{};
};

/*
struct event_info_debug {
  friend std::ostream& operator<<(std::ostream& out,
                                  event_info_debug const& eid) {
    auto const& sched = eid.sched_;
    auto const& ei = eid.ei_;

    auto const& st = sched.stations_[ei.station_->id_].get();
    out << "{st=" << st->eva_nr_ << " " << st->name_
        << ", i/o=" << ei.interchange_allowed_
        << ", sched_time=" << format_time(ei.schedule_time_)
        << ", cur_time=" << format_time(ei.current_time_)
        << ", reason=" << ei.timestamp_reason_
        << ", sched_track=" << ei.schedule_track_
        << ", cur_track=" << ei.current_track_;
    if (ei.ev_key_.is_not_null()) {
      auto const& ek = ei.ev_key_;
      out << ", type=" << (ek.is_departure() ? "DEP" : "ARR")
          << ", lcon_valid=" << ek.lcon_is_valid()
          << ", canceled=" << ek.is_canceled();
    }
    out << "}";
    return out;
  }

  schedule const& sched_;
  event_info const& ei_;
};
*/

}  // namespace

struct full_trip_handler {
  full_trip_handler(statistics& stats, schedule& sched,
                    update_msg_builder& update_builder,
                    message_history& msg_history, delay_propagator& propagator,
                    FullTripMessage const* msg,
                    std::string_view const msg_buffer,
                    std::map<schedule_event, delay_info*>& cancelled_delays,
                    rt_metrics& metrics, unixtime const msg_timestamp,
                    unixtime const processing_time)
      : stats_{stats},
        sched_{sched},
        update_builder_{update_builder},
        msg_history_{msg_history},
        propagator_{propagator},
        msg_{msg},
        msg_buffer_{msg_buffer},
        cancelled_delays_{cancelled_delays},
        metrics_{metrics},
        msg_timestamp_{msg_timestamp},
        processing_time_{processing_time} {}

  void handle_msg() {
    if (msg_->message_type() == FullTripMessageType_Schedule) {
      update_metrics([](metrics_entry* m) { ++m->ft_schedule_messages_; });
    } else {
      update_metrics([](metrics_entry* m) { ++m->ft_update_messages_; });
    }

    if (!check_events()) {
      LOG(warn) << "[FTH] invalid event sequence (status "
                << static_cast<int>(result_.status_) << ")";
      return;
    }

    get_or_add_station(msg_->trip_id()->start_station());
    get_or_add_station(msg_->trip_id()->target_station());
    auto const ftid = get_full_trip_id();
    auto const trip_uuid = parse_uuid(view(msg_->trip_id()->uuid()));
    result_.trp_ = find_existing_trip(ftid);
    result_.is_new_trip_ = result_.trp_ == nullptr;

    auto existing_sections = get_existing_sections(result_.trp_);
    auto sections = get_msg_sections();

    if (existing_sections.empty() && sections.empty()) {
      if (debug_) {
        LOG(info) << "[FTH] no sections, useless message";
      }
      return;
    }

    result_.is_reroute_ = !is_same_route(existing_sections, sections);

    if (debug_) {
      auto const rule_service = is_rule_service(result_.trp_);
      LOG(info) << "[FTH] is_new_trip=" << is_new_trip()
                << ", is_reroute=" << is_reroute()
                << ", is_rule_service=" << rule_service
                << ", existing_sections=" << existing_sections.size()
                << ", new_sections=" << sections.size()
                << ", trip_uuid=" << view(msg_->trip_id()->uuid());
    }

    if (is_reroute()) {
      auto const rule_service = is_rule_service(result_.trp_);
      if (is_new_trip()) {
        ++stats_.additional_msgs_;
        ++stats_.additional_total_;
        ++stats_.additional_ok_;
        update_metrics([](metrics_entry* m) { ++m->ft_new_trips_; });
      } else if (sections.empty()) {
        ++stats_.cancel_msgs_;
        update_metrics([](metrics_entry* m) { ++m->ft_cancellations_; });
      } else {
        ++stats_.reroute_msgs_;
        update_metrics([](metrics_entry* m) { ++m->ft_reroutes_; });
        if (!rule_service) {
          ++stats_.reroute_ok_;
        } else {
          update_metrics(
              [](metrics_entry* m) { ++m->ft_rule_service_reroutes_; });
        }
      }

      if (rule_service) {
        ++stats_.reroute_rule_service_not_supported_;
        if (debug_) {
          LOG(info) << "[FTH] ignoring message because reroute of rule "
                       "services is not supported";
        }
        return;
      }

      auto const canceled_ev_keys =
          get_canceled_ev_keys(existing_sections, sections);
      auto const old_trip_backup = get_trip_backup(result_.trp_);

      auto incoming = std::vector<incoming_edge_patch>{};
      save_outgoing_edges(get_station_nodes(sections), incoming);
      auto const lcs = build_light_connections(
          sections);  // TODO(pablo): reuse/copy existing
      auto const route = build_route(sections, lcs, incoming);
      patch_incoming_edges(incoming);
      result_.trp_ = create_or_update_trip(result_.trp_, ftid, route);
      update_delay_infos(existing_sections, sections);

      for (auto const station_idx : result_.stations_addded_) {
        update_builder_.add_station(station_idx);
      }

      if (old_trip_backup) {
        update_builder_.add_reroute(result_.trp_, old_trip_backup->edges_,
                                    old_trip_backup->lcon_idx_);
      } else {
        update_builder_.add_reroute(result_.trp_, {}, 0);
      }
      for (auto const& ev : canceled_ev_keys) {
        propagator_.recalculate(ev);
        if (auto const it = sched_.graph_to_delay_info_.find(ev);
            it != end(sched_.graph_to_delay_info_)) {
          cancelled_delays_.emplace(
              schedule_event{result_.trp_->id_.primary_, ev.get_station_idx(),
                             it->second->get_schedule_time(), ev.ev_type_},
              it->second);
        }
      }

      existing_sections = get_existing_sections(result_.trp_);
    }

    auto const recalculate_delays = is_reroute() && !is_new_trip();
    if (debug_) {
      LOG(info) << "[FTH] updating events, recalculate_delays="
                << recalculate_delays;
    }
    for (auto const& [msg_sec, cur_sec] :
         utl::zip(sections, existing_sections)) {
      update_event(cur_sec.dep_, msg_sec.dep_, cur_sec.lcon());
      update_event(cur_sec.arr_, msg_sec.arr_, cur_sec.lcon());
      if (recalculate_delays) {
        propagator_.recalculate(cur_sec.dep_.get_ev_key());
        propagator_.recalculate(cur_sec.arr_.get_ev_key());
      }
    }

    update_trip_uuid(trip_uuid);

    if (result_.delay_updates_ > 0) {
      ++stats_.delay_msgs_;
      update_metrics([this](metrics_entry* m) {
        ++m->ft_trip_delay_updates_;
        m->ft_event_delay_updates_ += result_.delay_updates_;
      });
    }
    if (result_.track_updates_ > 0) {
      ++stats_.track_change_msgs_;
      update_metrics([](metrics_entry* m) { ++m->ft_trip_track_updates_; });
    }
    msg_history_.add(result_.trp_->trip_idx_, msg_buffer_);
  }

private:
  station_node* get_or_add_station(StationInfo const* si) {
    if (auto const sn = find_station_node(sched_, view(si->eva()));
        sn != nullptr) {
      return sn;
    } else {
      return add_station(si);
    }
  }

  station_node* add_station(StationInfo const* si) {
    auto const index = static_cast<node_id_t>(sched_.stations_.size());
    utl::verify(index == sched_.station_nodes_.size(),
                "stations != station_nodes");
    utl::verify(index < sched_.non_station_node_offset_,
                "station limit reached");

    auto const st =
        sched_.stations_.emplace_back(mcd::make_unique<station>()).get();
    st->index_ = index;
    st->name_ = si->name()->str();
    st->eva_nr_ = si->eva()->str();
    st->transfer_time_ = 2;
    st->equivalent_.push_back(st);

    sched_.eva_to_station_.emplace(st->eva_nr_, st);
    if (si->ds100()->size() != 0) {
      sched_.ds100_to_station_.emplace(si->ds100()->str(), st);
    }

    auto& sn = sched_.station_nodes_.emplace_back(
        mcd::make_unique<station_node>(make_station_node(index)));
    constant_graph_add_station_node(sched_);

    result_.stations_addded_.emplace_back(index);

    return sn.get();
  }

  full_trip_id get_full_trip_id() const {
    full_trip_id ftid;

    auto const tid = msg_->trip_id()->id();
    ftid.primary_.station_id_ =
        get_station(sched_, view(tid->station_id()))->index_;
    ftid.primary_.time_ = unix_to_motistime(sched_, tid->time());
    ftid.primary_.train_nr_ = tid->train_nr();
    ftid.secondary_.target_station_id_ =
        get_station(sched_, view(tid->target_station_id()))->index_;
    ftid.secondary_.target_time_ =
        unix_to_motistime(sched_, tid->target_time());
    ftid.secondary_.line_id_ = view(tid->line_id());

    return ftid;
  }

  trip* find_existing_trip(full_trip_id const& ftid) {
    auto const first_it = std::lower_bound(
        begin(sched_.trips_), end(sched_.trips_),
        std::make_pair(ftid.primary_, static_cast<trip*>(nullptr)));
    if (first_it == end(sched_.trips_) || !(first_it->first == ftid.primary_)) {
      update_metrics([](metrics_entry* m) { ++m->ft_trip_id_not_found_; });
      return nullptr;
    }
    auto primary_matches = 0;
    for (auto it = first_it;
         it != end(sched_.trips_) && it->first == ftid.primary_; ++it) {
      if (it->second->id_.secondary_ == ftid.secondary_) {
        return it->second;
      }
      ++primary_matches;
    }
    if (primary_matches == 1) {
      // match by primary trip id for now if there is only one trip
      // with a matching primary trip id
      return first_it->second;
    } else if (primary_matches > 1) {
      update_metrics([](metrics_entry* m) { ++m->ft_trip_id_ambiguous_; });
    }
    return nullptr;
  }

  bool check_events() {
    auto last_arrival_station = std::string_view{};
    auto last_arrival_time = static_cast<time>(0);
    for (auto const& sec : *msg_->sections()) {
      auto const departure_station = view(sec->departure()->station()->eva());
      auto const arrival_station = view(sec->arrival()->station()->eva());
      if (!last_arrival_station.empty() &&
          last_arrival_station != departure_station) {
        result_.status_ = full_trip_result::status::INVALID_STATION_SEQUENCE;
        return false;
      }
      auto const departure_time =
          unix_to_motistime(sched_, sec->departure()->schedule_time());
      auto const arrival_time =
          unix_to_motistime(sched_, sec->arrival()->schedule_time());
      if (departure_time > arrival_time || departure_time < last_arrival_time) {
        result_.status_ =
            full_trip_result::status::INVALID_SCHEDULE_TIME_SEQUENCE;
        return false;
      }
      last_arrival_station = arrival_station;
      last_arrival_time = arrival_time;
    }
    return true;
  }

  std::vector<section> get_msg_sections() {
    return utl::to_vec(*msg_->sections(), [this](TripSection const* ts) {
      auto const from = get_or_add_station(ts->departure()->station());
      auto const to = get_or_add_station(ts->arrival()->station());
      return section{
          {from,
           ts->departure()->interchange_allowed(),
           unix_to_motistime(sched_, ts->departure()->schedule_time()),
           unix_to_motistime(sched_, ts->departure()->current_time()),
           from_fbs(ts->departure()->current_time_type()),
           get_track(sched_, view(ts->departure()->schedule_track())),
           get_track(sched_, view(ts->departure()->current_track())),
           {},
           parse_uuid(view(ts->departure()->uuid()))},
          {to,
           ts->arrival()->interchange_allowed(),
           unix_to_motistime(sched_, ts->arrival()->schedule_time()),
           unix_to_motistime(sched_, ts->arrival()->current_time()),
           from_fbs(ts->arrival()->current_time_type()),
           get_track(sched_, view(ts->arrival()->schedule_track())),
           get_track(sched_, view(ts->arrival()->current_track())),
           {},
           parse_uuid(view(ts->arrival()->uuid()))},
          nullptr};
    });
  }

  std::vector<section> get_existing_sections(trip const* trp) const {
    if (trp == nullptr) {
      return {};
    }
    return utl::to_vec(
        access::sections{trp}, [this, trp](access::trip_section const& sec) {
          auto const& lc = sec.lcon();
          auto const ev_from = sec.ev_key_from();
          auto const ev_to = sec.ev_key_to();
          auto const di_from = get_delay_info(sched_, ev_from);
          auto const di_to = get_delay_info(sched_, ev_to);
          return section{
              {sec.from_node()->get_station(), sec.from_node()->is_in_allowed(),
               di_from.get_schedule_time(), lc.d_time_, di_from.get_reason(),
               get_schedule_track(sched_, ev_from), sec.fcon().d_track_,
               ev_from, get_event_uuid(trp, ev_from)},
              {sec.to_node()->get_station(), sec.to_node()->is_out_allowed(),
               di_to.get_schedule_time(), lc.a_time_, di_to.get_reason(),
               get_schedule_track(sched_, ev_to), sec.fcon().a_track_, ev_to,
               get_event_uuid(trp, ev_to)},
              const_cast<light_connection*>(&lc)};  // NOLINT
        });
  }

  boost::uuids::uuid get_event_uuid(trip const* trp, ev_key const& evk) const {
    return motis::access::get_event_uuid(sched_, trp, evk)
        .value_or(boost::uuids::nil_uuid());
  }

  static bool is_rule_service(trip const* trp) {
    if (trp == nullptr) {
      return false;
    }

    auto const secs = access::sections{trp};
    return std::any_of(begin(secs), end(secs), [](auto const& sec) {
      return sec.lcon().full_con_->con_info_->merged_with_ != nullptr ||
             std::any_of(begin(sec.from_node()->incoming_edges_),
                         end(sec.from_node()->incoming_edges_),
                         [](edge const* e) {
                           return e->type() == edge::THROUGH_EDGE;
                         }) ||
             std::any_of(
                 begin(sec.to_node()->edges_), end(sec.to_node()->edges_),
                 [](edge const& e) { return e.type() == edge::THROUGH_EDGE; });
    });
  }

  std::vector<light_connection> build_light_connections(
      std::vector<section> const& sections) {
    auto lcs = std::vector<light_connection>{};
    lcs.reserve(sections.size());

    utl::verify(sections.size() == msg_->sections()->size(),
                "section size mismatch");
    for (auto const& [i, sec] : utl::enumerate(sections)) {
      auto const& ts = msg_->sections()->Get(i);
      auto lc = light_connection{};
      lc.valid_ = 1U;
      lc.d_time_ = sec.dep_.schedule_time_;
      lc.a_time_ = sec.arr_.schedule_time_;
      auto const con_info = get_con_info_with_provider(
          sched_, con_infos_, ts->category()->code()->view(),
          ts->line_id()->view(), ts->train_nr(), ts->provider()->code()->view(),
          ts->provider()->name()->view());
      // NOTE: connections are always created with track=schedule_track first.
      // if a different current_track is supplied, it will be updated in
      // update_event (so that both schedule + current track information is
      // stored).
      lc.full_con_ = get_full_con(sched_, con_info, sec.dep_.schedule_track_,
                                  sec.arr_.schedule_track_);
      lcs.emplace_back(lc);
    }
    return lcs;
  }

  static std::set<station_node*> get_station_nodes(
      std::vector<section> const& sections) {
    auto nodes = std::set<station_node*>{};
    for (auto const& sec : sections) {
      nodes.insert(sec.dep_.station_);
      nodes.insert(sec.arr_.station_);
    }
    return nodes;
  }

  std::vector<node*> build_route_nodes(
      std::vector<section> const& sections, std::uint32_t route_id,
      std::vector<incoming_edge_patch>& incoming) {
    assert(!sections.empty());
    auto nodes = std::vector<node*>{};
    nodes.reserve(sections.size() + 1);

    auto const build_node = [this, route_id, &incoming](station_node* sn,
                                                        bool in_allowed,
                                                        bool out_allowed) {
      return build_route_node(sched_, route_id, sched_.next_node_id_++, sn,
                              sched_.stations_.at(sn->id_)->transfer_time_,
                              in_allowed, out_allowed, incoming);
    };

    auto const& first_section = sections.front();
    nodes.emplace_back(build_node(first_section.dep_.station_,
                                  first_section.dep_.interchange_allowed_,
                                  true));
    for (auto i = 1ULL; i < sections.size(); ++i) {
      nodes.emplace_back(build_node(sections[i].dep_.station_,
                                    sections[i].dep_.interchange_allowed_,
                                    sections[i - 1].arr_.interchange_allowed_));
    }
    auto const& last_section = sections.back();
    nodes.emplace_back(build_node(last_section.arr_.station_, true,
                                  last_section.arr_.interchange_allowed_));

    assert(nodes.size() == sections.size() + 1);
    return nodes;
  }

  mcd::vector<trip::route_edge> build_route(
      std::vector<section>& sections, std::vector<light_connection> const& lcs,
      std::vector<incoming_edge_patch>& incoming) {
    if (sections.empty()) {
      return {};
    }
    auto trip_edges = mcd::vector<trip::route_edge>{};
    auto const route_id = sched_.route_count_++;

    auto const route_nodes = build_route_nodes(sections, route_id, incoming);

    for (auto i = 0ULL; i < sections.size(); ++i) {
      auto& sec = sections[i];  // NOLINT
      auto const& lc = lcs.at(i);

      // NOTE: at this point, the connections still have track=schedule_track,
      // but we connect the route nodes to the platform nodes for current_track.
      // the track inside the connections is updated later in update_event.

      auto const from_route_node = route_nodes[i];  // NOLINT
      auto const from_station_node = from_route_node->get_station();
      auto const from_station = sched_.stations_[from_station_node->id_].get();
      auto const from_platform =
          from_station->get_platform(sec.dep_.current_track_);

      auto const to_route_node = route_nodes[i + 1];  // NOLINT
      auto const to_station_node = to_route_node->get_station();
      auto const to_station = sched_.stations_[to_station_node->id_].get();
      auto const to_platform =
          to_station->get_platform(sec.arr_.current_track_);

      if (from_platform) {
        auto const pn = add_platform_enter_edge(
            sched_, from_route_node, from_station_node,
            from_station->platform_transfer_time_, from_platform.value());
        add_outgoing_edge(&pn->edges_.back(), incoming);
      }

      if (to_platform) {
        add_platform_exit_edge(sched_, to_route_node, to_station_node,
                               to_station->platform_transfer_time_,
                               to_platform.value());
        add_outgoing_edge(&to_route_node->edges_.back(), incoming);
      }

      auto const& route_edge = from_route_node->edges_.emplace_back(
          make_route_edge(from_route_node, to_route_node, {lc}));
      add_outgoing_edge(&route_edge, incoming);
      trip_edges.emplace_back(&route_edge);
      constant_graph_add_route_edge(sched_, &route_edge);

      sec.lc_ = const_cast<light_connection*>(  // NOLINT
          &route_edge.m_.route_edge_.conns_.front());
      sec.dep_.ev_key_ = ev_key{&route_edge, 0, event_type::DEP};
      sec.arr_.ev_key_ = ev_key{&route_edge, 0, event_type::ARR};
    }

    return trip_edges;
  }

  trip* create_or_update_trip(trip* trp, full_trip_id const& ftid,
                              mcd::vector<trip::route_edge> const& trip_edges) {
    std::optional<expanded_trip_index> old_eti;
    std::optional<expanded_trip_index> new_eti;

    auto const edges =
        sched_.trip_edges_
            .emplace_back(
                mcd::make_unique<mcd::vector<trip::route_edge>>(trip_edges))
            .get();

    auto const lcon_idx = static_cast<lcon_idx_t>(0U);

    if (trp == nullptr) {
      trp = sched_.trip_mem_
                .emplace_back(mcd::make_unique<trip>(
                    ftid, "", edges, lcon_idx,
                    static_cast<trip_idx_t>(sched_.trip_mem_.size()),
                    trip_debug{}, mcd::vector<uint32_t>{},
                    boost::uuids::nil_uuid()))
                .get();

      auto const trp_entry =
          mcd::pair{ftid.primary_, static_cast<ptr<trip>>(trp)};
      sched_.trips_.insert(
          std::lower_bound(begin(sched_.trips_), end(sched_.trips_), trp_entry),
          trp_entry);
    } else {
      old_eti = remove_expanded_trip(trp);
      for (auto const& e : *trp->edges_) {
        e.get_edge()->m_.route_edge_.conns_[trp->lcon_idx_].valid_ = 0U;
      }
      trp->edges_ = edges;
      trp->lcon_idx_ = lcon_idx;
    }

    if (!trip_edges.empty()) {
      // TODO(pablo): reuse existing
      auto const new_trps_id = sched_.merged_trips_.size();
      sched_.merged_trips_.emplace_back(
          mcd::make_unique<mcd::vector<ptr<trip>>,
                           std::initializer_list<ptr<trip>>>({trp}));

      for (auto const& trp_edge : trip_edges) {
        trp_edge.get_edge()->m_.route_edge_.conns_[lcon_idx].trips_ =
            new_trps_id;
      }

      new_eti = add_expanded_trip(trp);
    }

    update_builder_.expanded_trip_moved(trp, old_eti, new_eti);

    return trp;
  }

  std::optional<expanded_trip_index> remove_expanded_trip(trip const* trp) {
    if (trp->edges_->empty()) {
      return {};
    }
    auto const old_route_id = trp->edges_->front()->from_->route_;
    for (auto const old_exp_route_id :
         sched_.route_to_expanded_routes_.at(old_route_id)) {
      auto exp_route = sched_.expanded_trips_.at(old_exp_route_id);
      if (auto it = std::find(begin(exp_route), end(exp_route), trp);
          it != end(exp_route)) {
        auto const eti = expanded_trip_index{
            old_exp_route_id,
            static_cast<uint32_t>(std::distance(begin(exp_route), it))};
        exp_route.erase(it);
        if (exp_route.empty()) {
          utl::erase(sched_.route_to_expanded_routes_.at(old_route_id),
                     old_exp_route_id);
        }
        return eti;
      }
    }
    LOG(warn) << "rt::full_trip_handler: expanded trip not found: "
              << debug::trip{sched_, trp};
    return {};
  }

  expanded_trip_index add_expanded_trip(trip* trp) {
    assert(!trp->edges_->empty());
    auto const route_id = static_cast<uint32_t>(
        trp->edges_->front().get_edge()->get_source()->route_);
    return add_trip_to_new_expanded_route(sched_, trp, route_id);
  }

  void update_event(event_info const& cur_event, event_info const& msg_event,
                    light_connection* lc) {
    auto const& evk = cur_event.get_ev_key();
    if (cur_event.time_updated(msg_event)) {
      propagator_.add_delay(evk, msg_event.timestamp_reason_,
                            msg_event.current_time_);
      ++result_.delay_updates_;
    }
    if (cur_event.track_updated(msg_event)) {
      sched_.graph_to_schedule_track_index_.emplace(evk,
                                                    msg_event.schedule_track_);
      auto new_full_con = *lc->full_con_;
      if (evk.is_departure()) {
        new_full_con.d_track_ = msg_event.current_track_;
      } else {
        new_full_con.a_track_ = msg_event.current_track_;
      }
      lc->full_con_ =
          sched_.full_connections_
              .emplace_back(mcd::make_unique<connection>(new_full_con))
              .get();
      update_builder_.add_track_nodes(
          evk, sched_.tracks_.at(msg_event.current_track_).str(),
          msg_event.schedule_time_);
      ++result_.track_updates_;
    }
    auto const uuid_changed = cur_event.uuid_ != msg_event.uuid_;
    if (uuid_changed || is_reroute()) {
      if (uuid_changed && !cur_event.uuid_.is_nil()) {
        LOG(warn) << "event uuid changed: " << cur_event.uuid_ << " -> "
                  << msg_event.uuid_;
      }
      auto const trip_and_ev_key =
          mcd::pair{ptr<trip const>{result_.trp_}, evk};
      sched_.uuid_to_event_[msg_event.uuid_] = trip_and_ev_key;
      sched_.event_to_uuid_[trip_and_ev_key] = msg_event.uuid_;
    }
  }

  static std::vector<ev_key> get_canceled_ev_keys(
      std::vector<section> const& existing_sections,
      std::vector<section> const& new_sections) {
    auto canceled = std::vector<ev_key>{};
    for (auto const& ex_sec : existing_sections) {
      if (!std::any_of(begin(new_sections), end(new_sections),
                       [&](section const& new_sec) {
                         return new_sec.dep_.station_ == ex_sec.dep_.station_ &&
                                new_sec.dep_.schedule_time_ ==
                                    ex_sec.dep_.schedule_time_;
                       })) {
        canceled.emplace_back(ex_sec.dep_.get_ev_key());
      }
      if (!std::any_of(begin(new_sections), end(new_sections),
                       [&](section const& new_sec) {
                         return new_sec.arr_.station_ == ex_sec.arr_.station_ &&
                                new_sec.arr_.schedule_time_ ==
                                    ex_sec.arr_.schedule_time_;
                       })) {
        canceled.emplace_back(ex_sec.arr_.get_ev_key());
      }
    }
    return canceled;
  }

  void update_delay_infos(std::vector<section> const& existing_sections,
                          std::vector<section> const& new_sections) {
    if (existing_sections.empty() || new_sections.empty()) {
      return;
    }

    for (auto const& new_sec : new_sections) {
      update_delay_info(
          new_sec, existing_sections,
          [](section const& sec) -> event_info const& { return sec.dep_; });
      update_delay_info(
          new_sec, existing_sections,
          [](section const& sec) -> event_info const& { return sec.arr_; });
    }
  }

  template <typename GetEventInfo>
  void update_delay_info(section const& new_sec,
                         std::vector<section> const& existing_sections,
                         GetEventInfo&& get_ev) {
    auto const& new_ev = get_ev(new_sec);
    auto const& new_ev_key = new_ev.get_ev_key();
    if (auto const ex_sec =
            std::find_if(begin(existing_sections), end(existing_sections),
                         [&](section const& sec) {
                           auto const& ev = get_ev(sec);
                           return ev.station_ == new_ev.station_ &&
                                  ev.schedule_time_ == new_ev.schedule_time_;
                         });
        ex_sec != end(existing_sections)) {
      if (auto const di =
              sched_.graph_to_delay_info_.find(get_ev(*ex_sec).get_ev_key());
          di != end(sched_.graph_to_delay_info_)) {
        di->second->set_ev_key(new_ev_key);
        sched_.graph_to_delay_info_[new_ev_key] = di->second;
      }
    } else if (auto const di = cancelled_delays_.find(schedule_event{
                   result_.trp_->id_.primary_, new_ev.station_->id_,
                   new_ev.schedule_time_, new_ev_key.ev_type_});
               di != end(cancelled_delays_)) {
      di->second->set_ev_key(new_ev_key);
      sched_.graph_to_delay_info_[new_ev_key] = di->second;
    }
  }

  void update_trip_uuid(boost::uuids::uuid const& trip_uuid) {
    if (result_.trp_->uuid_ != trip_uuid) {
      if (!result_.trp_->uuid_.is_nil()) {
        LOG(warn) << "[FTH] trip uuid changed: " << result_.trp_->uuid_
                  << " => " << trip_uuid;
      }
      result_.trp_->uuid_ = trip_uuid;
      sched_.uuid_to_trip_[trip_uuid] = result_.trp_;
    }
  }

  static std::optional<trip_backup> get_trip_backup(trip const* trp) {
    if (trp != nullptr) {
      return trip_backup{*trp->edges_, trp->lcon_idx_};
    } else {
      return {};
    }
  }

  bool is_same_route_stop(event_info const& a, event_info const& b) {
    auto const st = sched_.stations_[a.station_->id_].get();
    return a.station_ == b.station_ &&
           a.interchange_allowed_ == b.interchange_allowed_ &&
           st->get_platform(a.current_track_) ==
               st->get_platform(b.current_track_);
  }

  bool is_same_route(std::vector<section> const& a,
                     std::vector<section> const& b) {
    if (a.size() != b.size()) {
      return false;
    }

    return std::all_of(begin(utl::zip(a, b)), end(utl::zip(a, b)),
                       [this](auto const& tup) {
                         auto const& [sa, sb] = tup;
                         return is_same_route_stop(sa.dep_, sb.dep_) &&
                                is_same_route_stop(sa.arr_, sb.arr_);
                       });
  }

  bool is_new_trip() const { return result_.is_new_trip_; }
  bool is_reroute() const { return result_.is_reroute_; }

  template <typename Fn>
  inline void update_metrics(Fn&& fn) {
    metrics_.update(msg_timestamp_, processing_time_, std::forward<Fn>(fn));
  }

public:
  full_trip_result get_result() const { return result_; }

  statistics& stats_;
  schedule& sched_;
  update_msg_builder& update_builder_;
  message_history& msg_history_;
  delay_propagator& propagator_;
  ris::FullTripMessage const* msg_;
  std::string_view msg_buffer_;
  full_trip_result result_;
  std::map<connection_info, connection_info const*> con_infos_;
  std::map<schedule_event, delay_info*>& cancelled_delays_;
  bool debug_{};
  rt_metrics& metrics_;
  unixtime msg_timestamp_;
  unixtime processing_time_;
};

full_trip_result handle_full_trip_msg(
    statistics& stats, schedule& sched, update_msg_builder& update_builder,
    message_history& msg_history, delay_propagator& propagator,
    ris::FullTripMessage const* msg, std::string_view const msg_buffer,
    std::map<schedule_event, delay_info*>& cancelled_delays,
    rt_metrics& metrics, unixtime const msg_timestamp,
    unixtime const processing_time) {
  auto handler = full_trip_handler{
      stats,      sched,         update_builder, msg_history,
      propagator, msg,           msg_buffer,     cancelled_delays,
      metrics,    msg_timestamp, processing_time};
  handler.handle_msg();
  return handler.get_result();
}

}  // namespace motis::rt
