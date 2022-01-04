#include "motis/loader/rule_route_builder.h"

#include <queue>

#include "utl/get_or_create.h"
#include "utl/pipes.h"

#include "motis/loader/rules_graph.h"
#include "motis/vector.h"

using namespace flatbuffers64;

namespace motis::loader {

struct rule_route_builder {
  rule_route_builder(graph_builder& gb,
                     bitfield const& schedule_traffic_days_mask)
      : gb_(gb), schedule_traffic_days_mask_(schedule_traffic_days_mask) {}

  void build(RuleService const* rs) {
    build_rules_graph(rs);
    build_routes();
  }

private:
  void build_rules_graph(RuleService const* rs) {
    std::map<Service const*, std::vector<service_node*>> service_nodes;

    auto const get_or_create_service_nodes = [&](Service const* s) {
      return utl::get_or_create(service_nodes, s, [&]() {
        auto utc_time_strings =
            gb_.service_times_to_utc(get_masked_traffic_days(s), s, true);
        if (!utc_time_strings.has_value()) {
          return std::vector<service_node*>{};
        }

        // Check for traffic days, where invalid times prevented
        // the conversion to UTC (indicated by an empty times vector).
        // Create these as unconnected service nodes, so they end up with
        // remaining bits and will be created as single trips.
        if (auto const invalid_it =
                utc_time_strings->find(mcd::vector<motis::time>{});
            invalid_it != end(*utc_time_strings)) {
          if (auto const fixed_invalid = gb_.service_times_to_utc(
                  invalid_it->second.local_traffic_days_, s, false);
              fixed_invalid.has_value()) {
            for (auto const& [times, traffic_days] : *fixed_invalid) {
              rg_.service_nodes_.emplace_back(
                  std::make_unique<service_node>(s, times, traffic_days));
            }
          }
          utc_time_strings->erase(invalid_it);
        }

        return utl::to_vec(*utc_time_strings,
                           [&](cista::pair<mcd::vector<time>,
                                           local_and_motis_traffic_days> const&
                                   utc_time_string) {
                             auto const& [times, traffic_days] =
                                 utc_time_string;
                             return rg_.service_nodes_
                                 .emplace_back(std::make_unique<service_node>(
                                     s, times, traffic_days))
                                 .get();
                           });
      });
    };

    for (auto const r : *rs->rules()) {
      if (skip_rule(r)) {
        continue;
      }

      auto const& s1_nodes = get_or_create_service_nodes(r->service1());
      auto const& s2_nodes = get_or_create_service_nodes(r->service2());

      for (auto* s1_node : s1_nodes) {
        for (auto* s2_node : s2_nodes) {
          auto const rn = rg_.rule_nodes_
                              .emplace_back(std::make_unique<rule_node>(
                                  s1_node, s2_node, r))
                              .get();
          s1_node->rule_nodes_.push_back(rn);
          s2_node->rule_nodes_.push_back(rn);
        }
      }
    }
  }

  bitfield get_masked_traffic_days(Service const* service) {
    return gb_.get_or_create_bitfield(service->traffic_days()) &
           schedule_traffic_days_mask_;
  }

  void build_routes() {
    for (auto& rn : rg_.rule_nodes_) {
      while (build_routes(rn.get())) {
      }
    }
    for (auto& sn : rg_.service_nodes_) {
      if (sn->traffic_days_.local_traffic_days_.any()) {
        left_over_trips_.emplace_back(sn.get());
      }
    }
  }

  bool build_routes(rule_node* ref_rn) {
    if (ref_rn->s1_->traffic_days_.local_traffic_days_.none() ||
        ref_rn->s2_->traffic_days_.local_traffic_days_.none()) {
      return false;
    }
    auto const ref_sn = ref_rn->s1_;
    auto ref_traffic_days = ref_sn->traffic_days_.local_traffic_days_;

    std::queue<std::tuple<rule_node*, service_node*, int>> queue;
    std::set<rule_node*> route_rules;
    std::map<service_node*, int> traffic_day_offsets;
    queue.emplace(ref_rn, ref_sn, 0);
    traffic_day_offsets[ref_sn] = 0;
    while (!queue.empty()) {
      auto const qe = queue.front();
      auto const [qrn, from, offset] = qe;
      queue.pop();

      auto const to = qrn->s1_ == from ? qrn->s2_ : qrn->s1_;
      auto const day_switch = qrn->rule_->day_switch();
      auto const delta_offset =
          qrn->s1_ == from
              ? qrn->rule_->day_offset2() - qrn->rule_->day_offset1() -
                    (day_switch ? 1 : 0)
              : qrn->rule_->day_offset1() - qrn->rule_->day_offset2() +
                    (day_switch ? 1 : 0);
      auto const new_offset = offset + delta_offset;
      auto const new_traffic_days =
          ref_traffic_days &
          shifted_bitfield(to->traffic_days_.local_traffic_days_, new_offset) &
          schedule_traffic_days_mask_;

      if (new_traffic_days.none()) {
        continue;
      }
      ref_traffic_days = new_traffic_days;
      route_rules.insert(qrn);
      traffic_day_offsets[to] = new_offset;

      for (auto sn : {qrn->s1_, qrn->s2_}) {
        for (auto rn : sn->rule_nodes_) {
          if (rn == qrn || route_rules.find(rn) != end(route_rules)) {
            continue;
          }
          auto const target_offset = sn == from ? offset : new_offset;
          if (rn->s1_ == sn) {
            queue.emplace(rn, rn->s1_, target_offset);
          } else if (rn->s2_ == sn) {
            queue.emplace(rn, rn->s2_, target_offset);
          }
        }
      }
    }

    if (ref_traffic_days.none() || route_rules.empty()) {
      return false;
    }

    auto& route = rule_routes_.emplace_back();
    for (auto const [sn, offset] : traffic_day_offsets) {
      auto const service_traffic_days =
          shifted_bitfield(ref_traffic_days, -offset) &
          schedule_traffic_days_mask_;
      route.traffic_days_[sn] =
          shifted_bitfield(service_traffic_days, sn->traffic_days_.shift_);
      assert(route.traffic_days_.at(sn).any());
      sn->traffic_days_.local_traffic_days_ &= ~service_traffic_days;
    }
    for (auto const& rn : route_rules) {
      route.rules_.push_back(rn);
    }
    return true;
  }

  inline bool skip_rule(Rule const* rule) const {
    return gb_.no_local_transport_ &&
           gb_.skip_route(rule->service1()->route()) &&
           gb_.skip_route(rule->service2()->route());
  }

public:
  graph_builder& gb_;
  rules_graph rg_;
  bitfield schedule_traffic_days_mask_;
  std::vector<service_node const*> left_over_trips_;
  mcd::vector<rule_route> rule_routes_;
};

void build_rule_routes(graph_builder& gb,
                       Vector<Offset<RuleService>> const* rule_services) {
  auto schedule_traffic_days_mask = create_uniform_bitfield('0');
  for (auto day_idx = gb.first_day_; day_idx <= gb.last_day_; ++day_idx) {
    if (day_idx >= schedule_traffic_days_mask.size()) {
      continue;
    }
    schedule_traffic_days_mask.set(day_idx, true);
  }

  rule_service_graph_builder rsgb(gb);

  for (auto const& rs : *rule_services) {
    rule_route_builder rrb(gb, schedule_traffic_days_mask);
    rrb.build(rs);

    for (auto const& sn : rrb.left_over_trips_) {
      gb.add_route_services({std::make_pair(
          sn->service_, sn->traffic_days_.local_traffic_days_)});
    }

    rsgb.add_rule_services(rrb.rule_routes_);
  }
}

}  // namespace motis::loader
