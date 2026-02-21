#include "motis/elevators/parse_siri_fm.h"

#include "pugixml.hpp"

namespace motis {

std::optional<elevator> parse_facility_condition(pugi::xml_node const& fc) {
  auto const id = fc.child_value("FacilityRef");
  if (id == nullptr || *id == '\0') {
    return std::nullopt;
  }

  auto const status = fc.child("FacilityStatus").child_value("Status");
  if (status == nullptr || *status == '\0') {
    return std::nullopt;
  }

  return elevator{0U,           std::string{id},
                  std::nullopt, std::string_view{status} == "available",
                  "",           {}};
}

vector_map<elevator_idx_t, elevator> parse_siri_fm(std::string_view s) {
  auto doc = pugi::xml_document{};
  if (!doc.load_buffer(s.data(), s.size())) {
    return {};
  }

  auto ret = vector_map<elevator_idx_t, elevator>{};
  for (auto fc : doc.child("Siri")
                     .child("ServiceDelivery")
                     .child("FacilityMonitoringDelivery")
                     .children("FacilityCondition")) {
    if (auto e = parse_facility_condition(fc); e.has_value()) {
      ret.emplace_back(std::move(*e));
    }
  }
  return ret;
}

vector_map<elevator_idx_t, elevator> parse_siri_fm(
    std::filesystem::path const& p) {
  return parse_siri_fm(
      cista::mmap{p.generic_string().c_str(), cista::mmap::protection::READ}
          .view());
}

}  // namespace motis
