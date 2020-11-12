#include "motis/loader/rule_route_builder.h"

#include "motis/vector.h"

#include <queue>

#include "utl/get_or_create.h"

using namespace flatbuffers64;

namespace motis::loader {

struct rule_node;

struct service_node {
  explicit service_node(Service const* service, bitfield&& traffic_days)
      : service_(service), traffic_days_(traffic_days) {}

  std::vector<rule_node*> rule_nodes_;
  Service const* service_{nullptr};
  bitfield traffic_days_;
};

struct rule_node {
  rule_node(service_node* s1, service_node* s2, Rule const* rule)
      : s1_(s1), s2_(s2), rule_(rule) {}

  service_node *s1_, *s2_;
  Rule const* rule_;
};

struct rules_graph {
  std::vector<std::unique_ptr<service_node>> service_nodes_;
  std::vector<std::unique_ptr<rule_node>> rule_nodes_;
};

struct rule_node_cmp {
  bool operator()(rule_node const* lhs, rule_node const* rhs) const {
    return lhs->rule_ < rhs->rule_;
  }
};

struct rule_route_builder {
  rule_route_builder(graph_builder& gb, bitfield const& traffic_days_mask)
      : gb_(gb), schedule_traffic_days_mask_(traffic_days_mask) {}

  void build(RuleService const* rs) {
    build_rules_graph(rs);
    build_routes();
  }

private:
  void build_rules_graph(RuleService const* rs) {
    std::map<Service const*, service_node*> service_to_node;
    for (auto const r : *rs->rules()) {
      if (skip_rule(r)) {
        continue;
      }
      auto s1_node = utl::get_or_create(service_to_node, r->service1(), [&]() {
        return rg_.service_nodes_
            .emplace_back(std::make_unique<service_node>(
                r->service1(), get_bitfield(r->service1())))
            .get();
      });
      auto s2_node = utl::get_or_create(service_to_node, r->service2(), [&]() {
        return rg_.service_nodes_
            .emplace_back(std::make_unique<service_node>(
                r->service2(), get_bitfield(r->service2())))
            .get();
      });
      auto rn =
          rg_.rule_nodes_
              .emplace_back(std::make_unique<rule_node>(s1_node, s2_node, r))
              .get();
      s1_node->rule_nodes_.push_back(rn);
      s2_node->rule_nodes_.push_back(rn);
    }
  }

  inline bitfield get_bitfield(Service const* service) {
    return deserialize_bitset<BIT_COUNT>(
               {service->traffic_days()->c_str(),
                static_cast<std::size_t>(service->traffic_days()->size())}) &
           schedule_traffic_days_mask_;
  }

  void build_routes() {
    for (auto& rn : rg_.rule_nodes_) {
      while (build_routes(rn.get())) {
      }
    }
    for (auto& sn : rg_.service_nodes_) {
      if (sn->traffic_days_.any()) {
        single_services_.emplace_back(sn->service_, sn->traffic_days_);
      }
    }
  }

  bool build_routes(rule_node* ref_rn) {
    if (ref_rn->s1_->traffic_days_.none() ||
        ref_rn->s2_->traffic_days_.none()) {
      return false;
    }
    auto const ref_sn = ref_rn->s1_;
    auto ref_traffic_days = ref_sn->traffic_days_;

    std::queue<std::tuple<rule_node*, service_node*, int>> queue;
    std::set<rule_node*, rule_node_cmp> route_rules;
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
          qrn->s1_ == from ? static_cast<int>(qrn->rule_->day_offset2()) -
                                 static_cast<int>(qrn->rule_->day_offset1()) -
                                 (day_switch ? 1 : 0)
                           : static_cast<int>(qrn->rule_->day_offset1()) -
                                 static_cast<int>(qrn->rule_->day_offset2()) +
                                 (day_switch ? 1 : 0);
      auto const new_offset = offset + delta_offset;
      auto const new_traffic_days =
          ref_traffic_days & shifted_bitfield(to->traffic_days_, new_offset) &
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
      route.traffic_days_[sn->service_] = service_traffic_days;
      sn->traffic_days_ &= ~service_traffic_days;
    }
    route.first_day_ = static_cast<unsigned>(gb_.first_day_);
    route.last_day_ = static_cast<unsigned>(gb_.last_day_);
    for (auto const& rn : route_rules) {
      route.rules_.push_back(rn->rule_);
    }
    return true;
  }

  static inline bitfield shifted_bitfield(bitfield const& orig, int offset) {
    return offset > 0 ? orig << static_cast<std::size_t>(offset)
                      : orig >> static_cast<std::size_t>(-offset);
  }

  inline bool skip_rule(Rule const* rule) {
    return gb_.no_local_transport_ &&
           (gb_.skip_route(rule->service1()->route()) ||
            gb_.skip_route(rule->service2()->route()));
  }

public:
  graph_builder& gb_;
  rules_graph rg_;
  bitfield const& schedule_traffic_days_mask_;
  std::vector<std::pair<Service const*, bitfield>> single_services_;
  mcd::vector<rule_route> rule_routes_;
};

void build_rule_routes(graph_builder& gb,
                       Vector<Offset<RuleService>> const* rule_services) {
  auto schedule_traffic_days_mask = create_uniform_bitfield<BIT_COUNT>('0');
  for (auto day_idx = gb.first_day_; day_idx <= gb.last_day_; ++day_idx) {
    schedule_traffic_days_mask.set(static_cast<std::size_t>(day_idx), true);
  }

  rule_service_graph_builder rsgb(gb);

  for (auto const& rs : *rule_services) {
    rule_route_builder rrb(gb, schedule_traffic_days_mask);
    rrb.build(rs);

    for (auto const& [service, traffic_days] : rrb.single_services_) {
      gb.add_route_services({std::make_pair(service, traffic_days)});
    }

    rsgb.add_rule_services(rrb.rule_routes_);
  }
}

}  // namespace motis::loader
