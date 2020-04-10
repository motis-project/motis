#pragma once

#include <functional>
#include <map>
#include <memory>
#include <set>
#include <vector>

#include "motis/loader/hrd/builder/station_builder.h"
#include "motis/loader/hrd/model/hrd_service.h"
#include "motis/loader/hrd/model/rule_service.h"
#include "motis/loader/hrd/model/service_rule.h"
#include "motis/loader/util.h"
#include "motis/schedule-format/RuleService_generated.h"

namespace motis::loader::hrd {

struct rule_service_builder {
  using service_builder_fun = std::function<flatbuffers64::Offset<Service>(
      hrd_service const&, bool, flatbuffers64::FlatBufferBuilder&)>;

  rule_service_builder() = default;
  explicit rule_service_builder(service_rules rs)
      : input_rules_(std::move(rs)) {}

  bool add_service(hrd_service const&);
  void resolve_rule_services();
  void create_rule_services(service_builder_fun const&, station_builder&,
                            flatbuffers64::FlatBufferBuilder&);

  std::vector<std::unique_ptr<hrd_service>> origin_services_;
  std::vector<rule_service> rule_services_;
  std::vector<flatbuffers64::Offset<RuleService>> fbs_rule_services_;

private:
  service_rules input_rules_;
};

}  // namespace motis::loader::hrd
