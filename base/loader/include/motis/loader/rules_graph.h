#pragma once

#include "motis/core/schedule/bitfield.h"
#include "motis/vector.h"

#include "motis/schedule-format/RuleService_generated.h"
#include "motis/schedule-format/Service_generated.h"

namespace motis::loader {

struct rule_node;

struct service_node {
  service_node(Service const* service, mcd::vector<time>&& times,
               bitfield const& traffic_days)
      : service_(service),
        times_(std::move(times)),
        traffic_days_(traffic_days) {}

  std::vector<rule_node*> rule_nodes_;
  Service const* service_{nullptr};
  mcd::vector<time> times_;
  bitfield traffic_days_;
};

struct rule_node {
  rule_node(service_node* s1, service_node* s2, Rule const* rule)
      : s1_(s1), s2_(s2), rule_(rule) {}

  friend bool operator<(rule_node const& a, rule_node const& b) {
    return a.rule_ < b.rule_;
  }

  service_node *s1_, *s2_;
  Rule const* rule_;
};

struct rules_graph {
  std::vector<std::unique_ptr<service_node>> service_nodes_;
  std::vector<std::unique_ptr<rule_node>> rule_nodes_;
};

}  // namespace motis::loader