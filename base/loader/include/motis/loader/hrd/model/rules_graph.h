#pragma once

#include <set>
#include <vector>

#include "motis/schedule/bitfield.h"
#include "motis/loader/hrd/model/rule_service.h"

namespace motis::loader::hrd {

struct rule_node;

struct node {
  node() = default;
  node(node const&) = default;
  node(node&&) = default;
  node& operator=(node const&) = default;
  node& operator=(node&&) = default;
  virtual ~node() = default;
};

struct service_node : public node {
  explicit service_node(hrd_service*);

  std::vector<rule_node*> rule_nodes_;
  hrd_service* service_{nullptr};
};

struct rule_node : public node {
  rule_node(service_node*, service_node*, resolved_rule_info);

  std::pair<std::set<rule_node*>, bitfield> max_component();
  void resolve_services(bitfield const& upper_traffic_days,
                        std::set<service_resolvent>& s_resolvents,
                        std::vector<service_rule_resolvent>& sr_resolvents);

  service_node *s1_, *s2_;
  resolved_rule_info rule_;
  bitfield traffic_days_;
};

struct rules_graph {
  std::vector<std::unique_ptr<node>> nodes_;
  std::vector<rule_node*> rule_nodes_;
};

}  // namespace motis::loader::hrd
