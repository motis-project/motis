#include "motis/loader/rule_service_graph_builder.h"

#include <algorithm>
#include <map>
#include <queue>
#include <set>
#include <vector>

#include "motis/core/common/logging.h"
#include "motis/core/schedule/price.h"
#include "motis/core/schedule/trip.h"
#include "motis/core/access/trip_iterator.h"
#include "motis/loader/util.h"
#include "utl/get_or_create.h"

#include "motis/schedule-format/Schedule_generated.h"

namespace motis::loader {

using namespace flatbuffers64;
using namespace motis::logging;

using neighbor = std::pair<Service const*, Rule const*>;

struct service_section {
  route_section route_section_;
  mcd::vector<participant> participants_;
  bool in_allowed_from_ = false;
  bool out_allowed_from_ = false;
  bool in_allowed_to_ = false;
  bool out_allowed_to_ = false;
};

struct rule_service_section_builder {
  rule_service_section_builder(graph_builder& gb, rule_route const& rs)
      : gb_(gb),
        neighbors_(build_neighbors(rs)),
        sections_(build_empty_sections(rs)) {}

  static std::map<Service const*, mcd::vector<neighbor>> build_neighbors(
      rule_route const& rs) {
    std::map<Service const*, mcd::vector<neighbor>> neighbors;
    for (auto const& r : rs.rules_) {
      if (r->type() != RuleType_THROUGH) {
        neighbors[r->service1()].emplace_back(r->service2(), r);
        neighbors[r->service2()].emplace_back(r->service1(), r);
      }
    }
    return neighbors;
  }

  static std::map<Service const*, mcd::vector<service_section*>>
  build_empty_sections(rule_route const& rs) {
    std::map<Service const*, mcd::vector<service_section*>> sections;
    for (auto const& r : rs.rules_) {
      sections.emplace(
          r->service1(),
          mcd::vector<service_section*>(
              static_cast<mcd::vector<service_section*>::size_type>(
                  r->service1()->sections()->size())));
      sections.emplace(
          r->service2(),
          mcd::vector<service_section*>(
              static_cast<mcd::vector<service_section*>::size_type>(
                  r->service2()->sections()->size())));
    }
    return sections;
  }

  static unsigned stop_index_of(Service const* s, Station const* station) {
    auto const& stations = *s->route()->stations();
    auto it = std::find(std::begin(stations), std::end(stations), station);
    utl::verify(it != std::end(stations), "rule station not found");
    return static_cast<unsigned>(std::distance(std::begin(stations), it));
  }

  void build_sections(rule_route const& rs) {
    std::set<Service const*> built;
    for (auto const& r : rs.rules_) {
      for (auto const& s : {r->service1(), r->service2()}) {
        if (built.insert(s).second) {
          add_service(s);
        }
      }
    }
  }

  void add_service(Service const* service) {
    add_service(service, 0, static_cast<unsigned>(service->sections()->size()),
                0, static_cast<unsigned>(service->sections()->size()),
                sections_[service], std::set<Service const*>());
  }

  void add_service(Service const* service,  //
                   unsigned from_idx, unsigned to_idx,  //
                   unsigned src_from_idx, unsigned src_to_idx,  //
                   mcd::vector<service_section*>& sections,
                   std::set<Service const*> visited) {
    visited.emplace(service);

    // Recursive add_service call for each neighbor.
    for (auto const& neighbor : neighbors_[service]) {
      Rule const* rule = neighbor.second;
      Service const* neighbor_service = neighbor.first;

      if (visited.find(neighbor_service) != end(visited)) {
        continue;
      }

      auto rule_service_from = stop_index_of(service, rule->from());
      auto rule_service_to = stop_index_of(service, rule->to());
      auto rule_neighbor_from = stop_index_of(neighbor_service, rule->from());
      auto rule_neighbor_to = stop_index_of(neighbor_service, rule->to());

      auto new_service_from_idx = std::max(rule_service_from, from_idx);
      auto new_service_to_idx = std::min(rule_service_to, to_idx);

      if (new_service_from_idx >= new_service_to_idx) {
        continue;
      }

      auto neighbor_from =
          rule_neighbor_from + (new_service_from_idx - rule_service_from);
      auto neighbor_to =
          rule_neighbor_to + (new_service_to_idx - rule_service_to);

      auto src_from = src_from_idx + (new_service_from_idx - from_idx);
      auto src_to = src_to_idx + (new_service_to_idx - to_idx);

      if (src_from < src_to) {
        add_service(neighbor_service, neighbor_from, neighbor_to, src_from,
                    src_to, sections, visited);
      }
    }

    // Add service as participant to the specified sections.
    for (unsigned src_section_idx = src_from_idx,
                  service_section_idx = from_idx;
         src_section_idx < src_to_idx;
         ++src_section_idx, ++service_section_idx) {
      if (sections[src_section_idx] == nullptr) {
        section_mem_.emplace_back(std::make_unique<service_section>());
        sections[src_section_idx] = section_mem_.back().get();
      }
      sections_[service][service_section_idx] = sections[src_section_idx];

      auto& section_participants = sections[src_section_idx]->participants_;
      auto not_already_added =
          std::find_if(begin(section_participants), end(section_participants),
                       [&service](participant const& p) {
                         return p.service_ == service;
                       }) == end(section_participants);

      if (not_already_added) {
        section_participants.emplace_back(service, service_section_idx);

        auto& ss = sections_[service][service_section_idx];
        ss->in_allowed_from_ |=
            service->route()->in_allowed()->Get(service_section_idx) != 0U;
        ss->out_allowed_from_ |=
            service->route()->out_allowed()->Get(service_section_idx) != 0U;
        ss->in_allowed_to_ |=
            service->route()->in_allowed()->Get(service_section_idx + 1) != 0U;
        ss->out_allowed_to_ |=
            service->route()->out_allowed()->Get(service_section_idx + 1) != 0U;
      }
    }
  }

  graph_builder& gb_;
  std::map<Service const*, mcd::vector<neighbor>> neighbors_;
  std::map<Service const*, mcd::vector<service_section*>> sections_;
  mcd::vector<std::unique_ptr<service_section>> section_mem_;
};

struct lcon_time_adjuster {
  static void adjust(edge* prev_edge, edge* e) {
    auto& prev_lcons = prev_edge->m_.route_edge_.conns_;
    auto& curr_lcons = e->m_.route_edge_.conns_;

    for (auto lcon_idx = lcon_idx_t{};
         lcon_idx < static_cast<int>(prev_lcons.size()); ++lcon_idx) {
      auto& prev_lcon = prev_lcons[lcon_idx];
      auto& curr_lcon = curr_lcons[lcon_idx];

      auto& last_arr = prev_lcon.a_time_;
      auto& curr_dep = curr_lcon.d_time_;
      auto& curr_arr = curr_lcon.a_time_;

      if (last_arr > curr_dep) {
        curr_dep += 60;
      }

      if (curr_dep > curr_arr) {
        curr_arr += 60;
      }

      assert(last_arr <= curr_dep && curr_dep <= curr_arr);
    }
  }

  void process_following_route_edges(edge* e, edge* pred) {
    for (auto& following : e->to_->edges_) {
      if (!following.empty()) {
        adjust(pred == nullptr ? e : pred, &following);
        queue_.emplace(&following, nullptr);
      } else if (following.type() == edge::THROUGH_EDGE) {
        queue_.emplace(&following, e);
      }
    }
  }

  void adjust_times(node* first_route_node) {
    for (auto& following : first_route_node->edges_) {
      queue_.emplace(&following, nullptr);
    }

    while (!queue_.empty()) {
      auto el = queue_.front();
      queue_.pop();
      process_following_route_edges(el.first, el.second);
    }
  }

  std::queue<std::pair<edge*, edge*>> queue_;
};

struct node_id_cmp {
  bool operator()(node const* lhs, node const* rhs) const {
    return lhs->id_ < rhs->id_;
  }
};

struct rule_service_route_builder {
  rule_service_route_builder(
      graph_builder& gb,  //
      unsigned first_day, unsigned last_day,
      std::map<Service const*, mcd::vector<service_section*>>& sections,
      unsigned route_id, rule_route const& rs)
      : gb_(gb),
        first_day_(first_day),
        last_day_(last_day),
        sections_(sections),
        route_id_(route_id),
        traffic_days_(rs.traffic_days_) {}

  void build_routes() {
    for (auto& entry : sections_) {
      build_route(entry.first, entry.second);
    }

    for (auto& entry : sections_) {
      write_trip_info(entry.first, entry.second);
      utl::verify(
          [&entry] {
            node const* pred_to = nullptr;
            for (auto const& s : entry.second) {
              if (pred_to != nullptr &&
                  pred_to != s->route_section_.from_route_node_) {
                return false;
              }
              pred_to = s->route_section_.to_route_node_;
            }
            return true;
          }(),
          "rule service: route node ordering");
    }
  }

  void adjust_times() {
    for (auto& entry : sections_) {
      lcon_time_adjuster().adjust_times(
          entry.second[0]->route_section_.from_route_node_);
    }
  }

  static services_key get_services_key(
      std::array<participant, 16> const& services, int day_idx) {
    services_key k;
    k.day_idx_ = day_idx;
    auto const ref_day_offset =
        services[0].service_->times()->Get(services[0].section_idx_ * 2 + 1) /
        1440;
    for (auto const& s : services) {
      if (s.service_ == nullptr) {
        break;
      }
      auto const s_day_offset =
          s.service_->times()->Get(s.section_idx_ * 2 + 1) / 1440;
      k.services_.insert({s.service_, ref_day_offset - s_day_offset});
    }
    return k;
  }

  merged_trips_idx get_or_create_trips(
      std::array<participant, 16> const& services, int day_idx) {
    auto k = get_services_key(services, day_idx);
    return utl::get_or_create(trips_, k, [&]() {
      return static_cast<merged_trips_idx>(push_mem(
          gb_.sched_.merged_trips_,
          mcd::to_vec(begin(k.services_), end(k.services_),
                      [&](service_with_day_offset sp) {
                        auto const s_day_idx = day_idx + sp.day_offset_;
                        return ptr<trip>{utl::get_or_create(
                            single_trips_,
                            std::make_pair(sp.service_, s_day_idx), [&]() {
                              return gb_.register_service(sp.service_,
                                                          s_day_idx);
                            })};
                      })));
    });
  }

  mcd::vector<light_connection> build_connections(
      service_section const& section, int third) {
    auto participants = section.participants_;
    std::sort(begin(participants), end(participants));

    std::array<participant, 16> services;
    std::copy(begin(participants), end(participants), begin(services));

    assert(!participants.empty());

    mcd::vector<light_connection> lcons;
    bool adjusted = false;
    auto const& traffic_days = traffic_days_.at(services[0].service_);
    for (unsigned day_idx = first_day_; day_idx <= last_day_; ++day_idx) {
      if (traffic_days.test(day_idx)) {
        lcons.push_back(
            gb_.section_to_connection(get_or_create_trips(services, day_idx),
                                      services, day_idx, 0, adjusted, third));
        adjusted = false;
      }
    }
    return lcons;
  }

  node* find_to(std::set<service_section*>& visited,
                mcd::vector<service_section*> const& sections, int i) const {
    if (i >= static_cast<int>(sections.size()) || i < 0 ||
        visited.find(sections[i]) != end(visited)) {
      return nullptr;
    }

    auto const& s = sections[i];
    visited.emplace(s);

    if (s->route_section_.to_route_node_ != nullptr) {
      return s->route_section_.to_route_node_;
    }

    auto const succ_from = find_from(visited, sections, i + 1);
    if (succ_from != nullptr) {
      return succ_from;
    }

    for (participant const& p : s->participants_) {
      auto const& p_sections = sections_.at(p.service_);
      auto const p_succ_from =
          find_from(visited, p_sections, p.section_idx_ + 1);
      if (p_succ_from != nullptr) {
        return p_succ_from;
      }
    }

    return nullptr;
  }

  node* find_from(std::set<service_section*>& visited,
                  mcd::vector<service_section*> const& sections, int i) const {
    if (i >= static_cast<int>(sections.size()) || i < 0 ||
        visited.find(sections[i]) != end(visited)) {
      return nullptr;
    }

    auto const& s = sections[i];
    visited.emplace(s);

    if (s->route_section_.from_route_node_ != nullptr) {
      return s->route_section_.from_route_node_;
    }

    auto const pred_to = find_to(visited, sections, i - 1);
    if (pred_to != nullptr) {
      return pred_to;
    }

    for (participant const& p : s->participants_) {
      auto const& p_sections = sections_.at(p.service_);
      auto const p_pred_to = find_to(visited, p_sections, p.section_idx_ - 1);
      if (p_pred_to != nullptr) {
        return p_pred_to;
      }
    }

    return nullptr;
  }

  void build_route(Service const* s, mcd::vector<service_section*>& sections) {
    utl::verify(s->sections()->size() == sections.size(),
                "section count mismatch");
    utl::verify(std::none_of(begin(sections), end(sections),
                             [](service_section* ss) { return ss == nullptr; }),
                "every section created");
    utl::verify(std::all_of(begin(sections), end(sections),
                            [&s](service_section const* ss) {
                              auto is_curr = [&s](participant const& p) {
                                return p.service_ == s;
                              };
                              auto contains_curr =
                                  std::find_if(begin(ss->participants_),
                                               end(ss->participants_),
                                               is_curr) !=
                                  end(ss->participants_);
                              return contains_curr;
                            }),
                "every section contains participant");
    utl::verify(
        std::all_of(begin(sections), end(sections),
                    [](service_section const* ss) {
                      auto const get_from_to = [](participant const& p) {
                        auto const stations = p.service_->route()->stations();
                        auto const from = stations->Get(p.section_idx_);
                        auto const to = stations->Get(p.section_idx_ + 1);
                        return std::make_pair(from, to);
                      };

                      auto const ref = get_from_to(ss->participants_.front());
                      return std::all_of(
                          begin(ss->participants_), end(ss->participants_),
                          [&get_from_to, &ref](participant const& p) {
                            return get_from_to(p) == ref;
                          });
                    }),
        "service section station mismatch");

    auto const& r = s->route();
    auto const& stops = r->stations();
    for (auto i = 0UL; i < stops->size() - 1; ++i) {
      auto section_idx = i;
      if (sections[section_idx]->route_section_.is_valid()) {
        continue;
      }

      auto from = section_idx;
      auto to = section_idx + 1;

      std::set<service_section*> v_from, v_to;  // visited sets
      auto const from_route_node = find_from(v_from, sections, section_idx);
      auto const to_route_node = find_to(v_to, sections, section_idx);

      int third = ((section_idx)*3) / stops->size();
      sections[section_idx]->route_section_ = gb_.add_route_section(
          route_id_, build_connections(*sections[section_idx], third),
          stops->Get(from),  //
          sections[section_idx]->in_allowed_from_,  //
          sections[section_idx]->out_allowed_from_,
          stops->Get(to),  //
          sections[section_idx]->in_allowed_to_,  //
          sections[section_idx]->out_allowed_to_,  //
          from_route_node, to_route_node);
    }

    utl::verify(
        std::all_of(begin(sections), end(sections),
                    [](auto&& s) { return s->route_section_.is_valid(); }),
        "all built sections are valid");
    utl::verify(
        [&sections]() {
          for (auto i = 0UL; i < sections.size() - 1; ++i) {
            if (sections[i]->route_section_.to_route_node_ !=
                sections[i + 1]->route_section_.from_route_node_) {
              return false;
            }
          }
          return true;
        }(),
        "all sections: s[i].to = s[i + 1].from");
  }

  void write_trip_info(Service const* s,
                       mcd::vector<service_section*> const& sections) {
    push_mem(
        gb_.sched_.trip_edges_,
        mcd::to_vec(
            begin(sections), end(sections), [](service_section* section) {
              return trip::route_edge(section->route_section_.get_route_edge());
            }));

    auto const& traffic_days = traffic_days_.at(s);
    auto edges = gb_.sched_.trip_edges_.back().get();
    auto lcon_idx = lcon_idx_t{};
    for (unsigned day_idx = first_day_; day_idx <= last_day_; ++day_idx) {
      if (traffic_days.test(day_idx)) {
        auto trp = single_trips_.at(std::make_pair(s, day_idx));
        trp->edges_ = edges;
        trp->lcon_idx_ = lcon_idx;
        ++lcon_idx;
      }
    }
  }

  void connect_through_services(rule_route const& rs) {
    for (auto const& r : rs.rules_) {
      if (r->type() == RuleType_THROUGH) {
        connect_route_nodes(r);
      }
    }
  }

  node* get_through_route_node(Service const* service, Station const* station,
                               bool source) {
    auto get_node = [source](service_section const* s) {
      return source ? s->route_section_.to_route_node_
                    : s->route_section_.from_route_node_;
    };

    auto station_it = gb_.stations_.find(station);
    utl::verify(station_it != end(gb_.stations_), "through station not found");
    auto station_node = station_it->second;

    auto& sections = sections_.at(service);
    if (source) {
      utl::verify(!sections.empty(),
                  "through station not found (sections empty)");
      auto const n = get_node(sections.back());
      utl::verify(n->get_station() == station_node,
                  "through station not found (last station does not match)");
      return n;
    } else {
      auto it = std::find_if(
          begin(sections), end(sections), [&](service_section const* s) {
            return get_node(s)->get_station() == station_node;
          });
      utl::verify(it != end(sections), "through station not found");
      return get_node(*it);
    }
  }

  void connect_route_nodes(Rule const* r) {
    auto s1_node = get_through_route_node(r->service1(), r->from(), true);
    auto s2_node = get_through_route_node(r->service2(), r->from(), false);
    for (auto const& e : s1_node->edges_) {
      if (e.type() == edge::THROUGH_EDGE && e.to_ == s2_node) {
        return;
      }
    }
    utl::verify(std::find_if(begin(s1_node->edges_), end(s1_node->edges_),
                             [](edge const& e) {
                               return e.type() == edge::THROUGH_EDGE;
                             }) == end(s1_node->edges_),
                "multiple outgoing through edges");
    s1_node->edges_.push_back(make_through_edge(s1_node, s2_node));
    through_target_nodes_.insert(s2_node);
  }

  void expand_trips() {
    for (auto& entry : sections_) {
      start_nodes_.insert(entry.second[0]->route_section_.from_route_node_);
    }
    for (auto const& entry : sections_) {
      for (auto const* section : entry.second) {
        auto const* rn = section->route_section_.to_route_node_;
        start_nodes_.erase(rn);
      }
    }
    for (auto const& n : through_target_nodes_) {
      start_nodes_.erase(n);
    }

    for (auto const* rn : start_nodes_) {
      expand_trips(rn);
    }
  }

  void expand_trips(node const* start_node) { expand_trips(start_node, {}); }

  void expand_trips(node const* start_node,
                    mcd::vector<trip::route_edge>&& prefix) {
    mcd::vector<trip::route_edge> path(std::move(prefix));
    auto n = start_node;
    while (n != nullptr) {
      mcd::vector<edge const*> route_edges;
      add_outgoing_route_edges(route_edges, n);
      if (route_edges.size() > 1) {
        for (auto const* re : route_edges) {
          auto new_prefix = path;
          new_prefix.emplace_back(re);
          expand_trips(re->to_, std::move(new_prefix));
        }
        return;
      } else if (route_edges.size() == 1) {
        auto const* re = route_edges[0];
        path.emplace_back(re);
        n = re->to_;
      } else {
        n = nullptr;
      }
    }
    make_expanded_trips(path);
  }

  void add_outgoing_route_edges(mcd::vector<edge const*>& route_edges,
                                node const* n) {
    for (auto const& e : n->edges_) {
      if (e.type() == edge::ROUTE_EDGE) {
        route_edges.push_back(&e);
      } else if (e.type() == edge::THROUGH_EDGE && e.to_ != n) {
        add_outgoing_route_edges(route_edges, e.to_);
      }
    }
  }

  void make_expanded_trips(mcd::vector<trip::route_edge>& route_edges) {
    if (route_edges.empty()) {
      return;
    }
    auto const lc_count = route_edges.front()->m_.route_edge_.conns_.size();

    push_mem(gb_.sched_.trip_edges_, route_edges);
    auto const edges_ptr = gb_.sched_.trip_edges_.back().get();

    auto trips_added = false;
    for (auto lcon_idx = 0U; lcon_idx < lc_count; ++lcon_idx) {
      full_trip_id ftid;
      push_mem(gb_.sched_.trip_mem_, ftid, edges_ptr, lcon_idx,
               static_cast<trip_idx_t>(gb_.sched_.trip_mem_.size()),
               trip_debug{}, mcd::vector<uint32_t>{});
      auto const trip_ptr = gb_.sched_.trip_mem_.back().get();
      if (gb_.check_trip(trip_ptr)) {
        gb_.sched_.expanded_trips_.push_back(trip_ptr);
        trips_added = true;
      }
    }
    if (trips_added) {
      gb_.sched_.expanded_trips_.finish_key();
    }
  }

  graph_builder& gb_;
  unsigned first_day_, last_day_;
  std::map<Service const*, mcd::vector<service_section*>>& sections_;
  std::map<std::pair<Service const*, int>, trip*> single_trips_;
  std::map<services_key, merged_trips_idx> trips_;
  unsigned route_id_;
  std::set<node const*, node_id_cmp> start_nodes_;
  std::set<node const*> through_target_nodes_;
  std::map<Service const*, bitfield> const& traffic_days_;
};

rule_service_graph_builder::rule_service_graph_builder(graph_builder& gb)
    : gb_(gb) {}

void rule_service_graph_builder::add_rule_services(
    mcd::vector<rule_route> const& rule_services) {
  if (rule_services.empty()) {
    return;
  }

  for (auto const& rule_service : rule_services) {
    auto route_id = gb_.next_route_index_++;

    rule_service_section_builder section_builder(gb_, rule_service);
    section_builder.build_sections(rule_service);

    rule_service_route_builder route_builder(
        gb_, rule_service.first_day_, rule_service.last_day_,
        section_builder.sections_, route_id, rule_service);
    route_builder.build_routes();
    route_builder.connect_through_services(rule_service);

    if (gb_.expand_trips_) {
      route_builder.expand_trips();
    }
  }
}

}  // namespace motis::loader
