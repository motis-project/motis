#include "./test_spec_test.h"

#include "motis/loader/hrd/hrd_parser.h"
#include "motis/loader/hrd/parser/service_parser.h"

namespace motis::loader::hrd {

using namespace flatbuffers64;

std::vector<specification> test_spec::get_specs() const {
  std::vector<specification> specs;
  parse_specification(
      lf_, [&specs](specification const& spec) { specs.push_back(spec); });
  return specs;
}

std::vector<hrd_service> test_spec::get_hrd_services(config const& c) const {
  std::vector<hrd_service> services;
  parse_specification(lf_, [&services, &c](specification const& spec) {
    services.emplace_back(hrd_service(spec, c));
  });
  return services;
}

}  // namespace motis::loader::hrd
