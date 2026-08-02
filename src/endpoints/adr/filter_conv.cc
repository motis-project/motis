#include "motis/endpoints/adr/filter_conv.h"

namespace a = adr;

namespace motis {

adr::filter_type to_filter_type(
    std::optional<std::vector<motis::api::LocationTypeEnum>> const& f) {
  if (!f.has_value() || f->empty()) {
    return a::filter_type::kNone;
  }
  auto result = a::filter_type::kNone;
  for (auto const t : *f) {
    switch (t) {
      case api::LocationTypeEnum::ADDRESS:
        result |= a::filter_type::kAddress;
        break;
      case api::LocationTypeEnum::PLACE:
        result |= a::filter_type::kPlace;
        break;
      case api::LocationTypeEnum::STOP: result |= a::filter_type::kExtra; break;
    }
  }
  return result;
}

}  // namespace motis