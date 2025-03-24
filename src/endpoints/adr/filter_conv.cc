#include "motis/endpoints/adr/filter_conv.h"

namespace a = adr;

namespace motis {

adr::filter_type to_filter_type(
    std::optional<motis::api::LocationTypeEnum> const& f) {
  if (f.has_value()) {
    switch (*f) {
      case api::LocationTypeEnum::ADDRESS: return a::filter_type::kAddress;
      case api::LocationTypeEnum::PLACE: return a::filter_type::kPlace;
      case api::LocationTypeEnum::STOP: return a::filter_type::kExtra;
    }
  }
  return a::filter_type::kNone;
}

}  // namespace motis