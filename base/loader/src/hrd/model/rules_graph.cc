#include "motis/loader/hrd/model/rules_graph.h"
#include "motis/schedule-format/RuleService_generated.h"

namespace motis::loader::hrd {

hrd_service* resolve(bitfield const& upper_traffic_days, hrd_service* origin,
                     std::set<service_resolvent>& resolved_services) {
  auto resolved_it = resolved_services.find(service_resolvent(origin));
  if (resolved_it == end(resolved_services)) {
    auto resolved = std::make_unique<hrd_service>(*origin);
    resolved->traffic_days_ &= upper_traffic_days;
    origin->traffic_days_ &= ~upper_traffic_days;
    std::tie(resolved_it, std::ignore) =
        resolved_services.emplace(std::move(resolved), origin);
  }
  return resolved_it->service_.get();
}

void rule_node::resolve_services(
    bitfield const& upper_traffic_days,
    std::set<service_resolvent>& s_resolvents,
    std::vector<service_rule_resolvent>& sr_resolvents) {
  if (traffic_days_.any()) {
    auto const active_traffic_days = traffic_days_ & upper_traffic_days;
    traffic_days_ &= ~active_traffic_days;
    auto const s1_traffic_days_offset =
        rule_.s1_traffic_days_offset_ + (rule_.day_switch_ ? 1 : 0);
    auto const s1_traffic_days = active_traffic_days >> s1_traffic_days_offset;
    auto const s2_traffic_days =
        active_traffic_days >> rule_.s2_traffic_days_offset_;
    sr_resolvents.emplace_back(
        rule_,  //
        resolve(s1_traffic_days, s1_->service_, s_resolvents),
        resolve(s2_traffic_days, s2_->service_, s_resolvents));
  }
}

service_node::service_node(hrd_service* s) : service_(s) {}

rule_node::rule_node(service_node* s1, service_node* s2,
                     resolved_rule_info rule_info)
    : s1_(s1), s2_(s2), rule_(rule_info) {
  switch (rule_.type_) {
    case RuleType_MERGE_SPLIT:
      rule_.s1_traffic_days_offset_ = s1->service_->traffic_days_offset_at_stop(
          s1->service_->get_first_stop_index_at(rule_info.eva_num_1_),
          event_type::DEP);
      rule_.s2_traffic_days_offset_ = s2->service_->traffic_days_offset_at_stop(
          s2->service_->get_first_stop_index_at(rule_info.eva_num_1_),
          event_type::DEP);
      break;
    case RuleType_THROUGH:
      rule_.s1_traffic_days_offset_ = s1->service_->traffic_days_offset_at_stop(
          s1->service_->stops_.size() - 1, event_type::ARR);
      rule_.s2_traffic_days_offset_ = s2->service_->traffic_days_offset_at_stop(
          s2->service_->get_first_stop_index_at(rule_info.eva_num_1_),
          event_type::DEP);
      break;
    default: throw std::runtime_error("unknown rule type");
  }
  auto const s1_mask_traffic_days_offset =
      rule_.s1_traffic_days_offset_ + (rule_.day_switch_ ? 1 : 0);
  traffic_days_ =
      (s1->service_->traffic_days_ << s1_mask_traffic_days_offset) &
      (s2->service_->traffic_days_ << rule_.s2_traffic_days_offset_) &
      rule_info.traffic_days_;
}

std::pair<std::set<rule_node*>, bitfield> rule_node::max_component() {
  std::pair<std::set<rule_node*>, bitfield> max;
  auto& component_nodes = max.first;
  auto& component_traffic_days = max.second;

  rule_node* current = nullptr;
  std::set<rule_node*> queue = {this};
  component_traffic_days = create_uniform_bitfield<BIT_COUNT>('1');
  while (!queue.empty()) {
    auto first_element = queue.begin();
    current = *first_element;
    queue.erase(first_element);

    auto next_traffic_days = component_traffic_days & current->traffic_days_;
    if (next_traffic_days.none()) {
      continue;
    }
    component_traffic_days = next_traffic_days;
    component_nodes.insert(current);

    for (auto const& link_node : {current->s1_, current->s2_}) {
      for (auto const& related_node : link_node->rule_nodes_) {
        if (related_node != current &&
            component_nodes.find(related_node) == end(component_nodes)) {
          queue.insert(related_node);
        }
      }
    }
  }

  return max;
}

}  // namespace motis::loader::hrd
