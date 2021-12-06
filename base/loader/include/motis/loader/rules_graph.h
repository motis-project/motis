#pragma once

#include "motis/core/schedule/bitfield.h"
#include "motis/vector.h"

#include "motis/loader/local_and_motis_traffic_days.h"

#include "motis/schedule-format/RuleService_generated.h"
#include "motis/schedule-format/Service_generated.h"

namespace motis::loader {

struct rule_node;

struct service_node {
  service_node(Service const* service, mcd::vector<time> times,
               local_and_motis_traffic_days const& local_traffic_days)
      : service_(service),
        times_(std::move(times)),
        traffic_days_(local_traffic_days) {
    assert(times_.size() == service->sections()->size() * 2);
  }

  std::vector<rule_node*> rule_nodes_;
  Service const* service_{nullptr};
  mcd::vector<time> times_;
  local_and_motis_traffic_days traffic_days_;
};

struct rule_node {
  rule_node(service_node* s1, service_node* s2, Rule const* rule)
      : s1_(s1), s2_(s2), rule_(rule) {}

  friend bool operator<(rule_node const& a, rule_node const& b) {
    return a.rule_ < b.rule_;
  }

  friend std::ostream& operator<<(std::ostream& out, rule_node const& rn) {
    auto const print_service = [&](service_node const* const s) {
      out << "  train_nr=" << s->service_->sections()->Get(0)->train_nr()
          << " ";
      print(std::cerr, s->traffic_days_.local_traffic_days_);
      out << ", offset=" << s->traffic_days_.shift_ << "\n";
      const auto stations = s->service_->route()->stations();
      for (auto i = 0U; i != stations->size(); ++i) {
        out << "     " << stations->Get(i)->name()->str() << " "
            << (i == 0U ? "       " : format_time(s->times_.at(2 * i - 1)))
            << (i != 0U && i != stations->size() - 1 ? " - " : "   ")
            << (i == stations->size() - 1 ? ""
                                          : format_time(s->times_.at(2 * i)))
            << "\n";
      }
    };

    std::cerr << EnumNameRuleType(rn.rule_->type()) << ": "
              << rn.rule_->from()->name()->str() << " - "
              << rn.rule_->to()->name()->str()
              << " day_offset1=" << rn.rule_->day_offset1()
              << ", day_offset2=" << rn.rule_->day_offset2()
              << ", day_switch=" << (rn.rule_->day_switch() ? "true" : "false")
              << "\n";

    print_service(rn.s1_);
    print_service(rn.s2_);

    return out;
  }

  service_node *s1_, *s2_;
  Rule const* rule_;
};

struct rules_graph {
  std::vector<std::unique_ptr<service_node>> service_nodes_;
  std::vector<std::unique_ptr<rule_node>> rule_nodes_;
};

}  // namespace motis::loader
